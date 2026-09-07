---
title: "Public docs + API-surface audit — spec"
type: spec
status: approved-for-implementation
created: 2026-09-06
updated: 2026-09-07
tier: high-stakes
reviewed: 2026-09-07 (adversarial specflow:review — 2 Blockers found and resolved
  below, 5 Medium + 5 Low findings folded in; see Review disposition section)
refs:
  - docs/api.md
  - .ai-docs/specs/public-docs-and-api-surface-audit-spec-2026-09-06.md
  - .ai-docs/adrs/adr-053-ts-js-mcp-distribution-shape-2026-06-24.md
  - .ai-docs/research/competitive-landscape/write-path-teardown-2026-09-06.md
  - .ai-docs/research/steal-matrix-rescore-2026-07-27.md
  - .ai-docs/tech-debt/tech-debt-register.md
---

# Public docs + API-surface audit

> Relocated/reformatted into specflow convention from
> `.ai-docs/specs/public-docs-and-api-surface-audit-spec-2026-09-06.md` (the
> original narrative doc — kept, not deleted, and now marked superseded-in-place
> for provenance). All content below is drawn from that doc verbatim where it is
> WHAT/WHY; the phased HOW moves to `plan.md` in this same directory.

## Why now, and why in this order

The ask was: build a Docusaurus site for the public docs, write getting-started
material, and use the exercise to find API discrepancies.

**The stated precondition is already met.** The ask was gated on *"once all
these TD items and missing features from our steal list [are closed]"*.
Measured 2026-09-06/07: the write-path half of the "steal list"
(`write-path-teardown-2026-09-06.md`) is CLOSED. The recall-scoring half
(`steal-matrix-rescore-2026-07-27.md`) had 6 of 8 items open when this spec's
narrative predecessor was first drafted — **re-verified 2026-09-07: now 4 of
8**, since 2 of those 6 (the FTS5 porter-stemmer tokenizer, and the RRF `k`
default flip 60→1) shipped in commits `6412c0cc` and `1ed3f506`, both landing
in this same working session, *after* the original draft's count was written
and carried forward unchanged into this reformatted spec on first pass — caught
by this spec's own adversarial review (below), which is itself proof that even
a number re-verified once can decay within the same session and needs
re-checking at point of use, not point of authorship. The 4 still open (mem0-style
entity round-trip, candidate-pool expansion, `entity_stream_weight`, TD-140
overfetch) do not block the docs/audit work, which is a P4-shaped best-effort
concern per the readiness roadmap's own tiering, not a correctness gate. The 65
open items in the TD register predate this session and are gating on a number
that has never been zero.

**Every measured count in this section should be re-verified at Phase 0
kickoff, not just the 69-block figure below** — this session already
demonstrated that a count can go stale between authorship and use.

**The audit is the deliverable; the docs are the instrument.** Writing a
getting-started guide is the highest-yield API bug-finder available, because it
forces walking the consumer's exact path with none of the maintainer's context.
Proven twice already: the v0.3.0 consumer-DX audit found the crate not
installable out of the box; TD-231 existed because a doc comment pointed at "the
builder" for a setter that did not exist, unreachable for months. The docs are
the **probe**; the API fixes are the **output**.

**The site is blocked; the content is not.** Docusaurus deploy needs the repo
public (free CI) and a DNS record on `kremory.dev` — both standing
maintainer-only decisions, neither blocks the content work. Split them.

## The measurement that makes this rigorous

Counted 2026-09-06 (re-verify at Phase 0 kickoff — the doc surface may have
moved since):

| | count |
|---|---|
| Rust code blocks in public docs (`docs/*.md` + `README.md`) | 69 (46 in `api.md`, 9 in README) |
| …of those, compiled by anything today | 0 |
| Doctests currently passing | 26 — but these cover only inline `///` comments |
| TypeScript / JavaScript examples anywhere in the docs | 0 |

69 uncompiled examples is the discrepancy list — extracted and compiled against
the real crate, every failure is a concrete drift between what is documented and
what ships. The Node binding has zero examples in the language its users write;
that is a product gap, not a docs gap.

## Scope

### Surfaces, and their honest status

| Surface | Ships today | Documentation stance |
|---|---|---|
| Rust crate (`kremory` on crates.io) | ✅ 0.7.0 live (re-verified 2026-09-07 via crates.io API), 0.7.1 prepared, unpublished | Primary. Document as shipping. |
| Node binding (`@kgentic-ai/kremory-node`) | ❌ 0.4.1, unpublished | Document, but label **unpublished** explicitly |
| MCP server (`@kgentic-ai/kremory-mcp`) | ❌ does not exist (ADR-053) | Out of scope for content; note as planned |

## Behaviour rules

- **RULE-001**: WHEN a Rust code block in `docs/*.md` or `README.md` has no
  explicit non-compiling marker (`no_run`, `compile_fail`, "illustrative",
  "pseudo-code"), the system SHALL compile it successfully against the crate
  under audit (workspace HEAD for Phase 0; the published crate specifically for
  Phase 3's getting-started transcript). **Compile ≠ run**: Phase 0 is a
  type-check only; examples needing a live LLM/embedder connection to execute
  are marked `no_run` and are compiled, never executed, by the harness. **Scaffold
  contract (added post-review — Blocker 1)**: most of the 69 existing blocks are
  narrative fragments, not standalone programs — bare top-level `.await?`
  statements, or references to undeclared placeholder bindings (`my_llm`,
  `my_embedder`, `mem`, `MySink`) that only make sense read in the prose
  surrounding them. The harness SHALL compile blocks **per-section** (all
  fenced blocks under one `##`/`###` heading, concatenated in document order,
  share one compilation unit) rather than per-block in isolation, AND SHALL
  inject a documented prelude of standard placeholder bindings (at minimum:
  `my_llm: Arc<dyn ChatProvider>`, `my_embedder: Arc<dyn DynEmbeddingProvider>`,
  a `MySink` unit struct implementing `EnrichmentEventSink`) so that narrative
  examples referencing them compile without each doc author having to make
  every fragment self-contained. Phase 0's own task breakdown SHALL name the
  exact prelude contents before extraction starts — this is a design decision,
  not an implementation detail, and gets it wrong once and silently mislabels
  ~65 of 69 examples as broken.
- **RULE-002**: WHEN the compile-audit harness runs, IF it extracts fewer than
  60 code blocks, THE harness SHALL fail loudly (vacuous-pass guard — this
  failure mode has already occurred twice in this repo: the `consistency_check`
  guard and the ADR-integrity guard's own scan set). 60 ≈ 87% of the current
  69-block count — chosen to tolerate minor doc churn while still catching a
  broken or near-empty extractor.
- **RULE-012** (added post-review — Blocker 2): WHERE a RULE-006 fix is
  implemented and committed locally but NOT published (a breaking, unpublished
  change), every doc example whose correctness depends on that fix SHALL be
  labelled with its target version ("as of vX.Y, unreleased — see Phase 2
  finding N") until the fix is published on crates.io. This applies to every
  in-scope doc (`api.md`, `README.md`, `observability.md`, etc.), not only the
  getting-started guide (RULE-008 already pins that one to the published
  crate). Without this rule, an unpublished Phase-2 fix would make the rest of
  the audited docs describe capabilities a real `cargo add kremory` consumer
  does not have — reproducing the exact TD-231 "documented but absent" defect
  class this audit exists to eliminate, self-inflicted by the audit's own fix
  path.
- **RULE-013** (added post-review — Finding 5): WHERE a Phase 2 fix requires a
  new `kremory-napi` parity-skip-list entry, IF the skip-list is already at its
  enforced cap (verified 2026-09-07: 90/90, `crates/kremory-napi/tests/api_parity.rs:502`),
  raising the cap by exactly the number of new entries, with a one-line
  rationale per the file's own established raise-with-history convention, IS
  in scope for this spec. If a finding instead requires a full napi mirror
  (not just a skip-list entry), that is also in scope but SHALL be called out
  explicitly as larger-than-typical Phase 2 work.
- **RULE-003**: WHEN a public, reachable API item exists on an in-scope surface,
  the documentation SHALL mention it (no undocumented-and-unmentioned surface).
- **RULE-004**: WHEN documentation describes a method, field, or builder call,
  IF that exact call does not exist or is not reachable through any public
  path, THEN it is a defect of class "documented-but-absent" or "unreachable"
  (the TD-231 class) and SHALL be fixed per RULE-006.
- **RULE-005**: WHEN documentation describes API signature, default value, or
  behaviour, IF the actual code differs, THEN it is a defect of class "wrong"
  and SHALL be fixed per RULE-006.
- **RULE-006**: WHEN a Phase 1 finding requires a fix, the system SHALL fix it
  at the layer that is actually wrong (code OR doc, decided per-finding, stated
  before the fix) — never reconcile a doc to broken behaviour, never change
  behaviour to match an aspirational doc. WHERE the correct fix is a breaking
  API change, the change SHALL be implemented and committed locally but SHALL
  NOT be published (consistent with the standing `cargo publish` HITL gate) —
  flagged explicitly in the Phase 2 findings table for the maintainer's
  go/no-go. **Disambiguated post-review (Finding 7)**: "the Phase 2 findings
  table" IS the SC-003 semantic-audit table, with a Fixed/NOT-FIXED column
  added — not a separate artifact.
- **RULE-007**: WHERE a package is unpublished (`@kgentic-ai/kremory-node`) or
  nonexistent (the MCP server), documentation SHALL label it explicitly as such
  and SHALL NOT imply it ships today.
- **RULE-008**: WHEN the getting-started guide is written, it SHALL be
  validated by actually running `cargo add kremory` against the published
  crate (0.7.0) in a fresh directory with no repo checkout, and the doc SHALL
  match that real transcript.
- **RULE-009**: WHEN the Docusaurus site is built locally (`npm run build`), it
  SHALL exit 0, every existing public doc SHALL be reachable from the nav, and
  there SHALL be zero dead internal links.
- **RULE-010**: The Docusaurus site SHALL NOT be deployed as part of this spec
  — deploy is blocked on the maintainer's repo-public + DNS decisions and is
  listed for visibility only.
- **RULE-011**: WHERE the Node binding documents a capability that is not
  labelled planned/unpublished, at least one JS/TS example SHALL exist for it
  AND SHALL actually execute (or, where execution needs a live LLM/embedder,
  SHALL at minimum type-check via `tsc` against `index.d.ts`) against the
  locally-built native module — **strengthened post-review (Finding 11)**: the
  original wording only required the example to "exist," which would let the
  Node surface — the exact surface this spec calls out as having zero examples
  today — inherit the same "written but never verified" defect the Rust
  compile-audit (RULE-001/002) exists to catch. `kremory-napi/__test__/*.test.mjs`
  and `types.check.ts` are the existing local harness this rule routes through;
  no new test infrastructure is required, only new example content run
  through it.

## Success criteria

- **SC-001**: Every Rust doc code block (freshly re-counted at Phase 0 kickoff)
  is either (a) compiling, or (b) explicitly marked non-compiling with a stated
  reason — zero unmarked failures.
- **SC-002**: Compile-audit harness asserts extracted-block count ≥ 60 and
  fails the build if the assertion does not hold.
- **SC-003**: A semantic-audit table exists covering all four axes
  (Undocumented / Documented-but-absent / Unreachable / Wrong) with `file:line`
  cited on both the code side and the doc side for every row.
- **SC-004**: Every row in SC-003's table is either fixed (with the fixed layer
  named) or explicitly recorded **NOT FIXED** with a reason — no silent drops.
- **SC-005**: `npm run build` (Docusaurus) exits 0; 100% of existing docs are
  reachable from nav; 0 dead internal links.
- **SC-006**: RULE-011 holds for every in-scope, non-deferred Node capability.
- **SC-007**: The getting-started doc has a captured real-run transcript and
  the doc text matches it exactly; any friction found during the run is
  recorded as an SC-004 row, not silently smoothed over in prose.
- **SC-008**: `/quality-gate` (typecheck/lint/test/build) is green after every
  Phase 2 code-side fix, before that fix is committed.
- **SC-009** (covers RULE-007, added during self-`analyze` — the first draft
  had no SC mapped to it): every documentation reference to
  `@kgentic-ai/kremory-node` or the MCP server explicitly states its
  non-shipping status (unpublished / does not exist) — zero
  implied-shipping references.
- **SC-010** (added post-review — Finding 6): every Phase-0 compile failure
  resolved via a non-compiling marker (`no_run`/`compile_fail`) rather than a
  code/doc fix ALSO appears as a row in the SC-003 table with its axis
  classification and reason — a marker alone, with no SC-003 row, does not
  satisfy SC-001. Closes the loophole where a real "wrong"-class defect gets
  silently marked away instead of triaged, which is exactly the failure mode
  this audit exists to catch, one layer down inside its own tooling.
- **SC-011** (added post-review — Finding 12): every `[dependencies]`/TOML
  version pin and every version banner in an in-scope doc matches the
  currently-documented target version (the published crate for anything
  RULE-008-scoped, workspace HEAD otherwise) — covers a defect class
  (stale `version = "0.6"` pins, a stale `v0.6.0` banner) that sits outside
  both RULE-001's compile-audit (not inside a ` ```rust ` fence) and the
  Phase 1 four-axis table as originally scoped (framed around API items, not
  version strings). Verified live 2026-09-07: 4 stale pins + 1 stale banner
  already present in `docs/api.md`, `docs/observability.md`, `README.md`.

## Non-goals

- Deploying the Docusaurus site (blocked; RULE-010).
- The marketing site (explicitly deferred by the maintainer to a later,
  separate piece of work).
- The TypeScript MCP wrapper build (ADR-053; separate spec).
- Resolving any of the 65 pre-existing open tech-debt items, except ones this
  audit independently surfaces as new findings. **Disambiguated post-review
  (Finding 8)**: a Phase-1 finding is in-scope for fixing under this spec
  regardless of whether a TD entry already exists for it, provided the fix is
  small/local (comparable effort to an existing sibling-pattern fix, e.g.
  TD-231's accessor addition). A finding that maps to a TD requiring larger,
  independently-scoped work gets a NOT-FIXED row in the SC-003 table citing the
  TD number, not silent exclusion.
- Re-litigating the 3 recall-scoring levers already deliberately held per the
  project's own hold-off rule (mem0-style entity round-trip, candidate-pool
  expansion, TD-140 overfetch fix) — untouched by this spec.
- Docusaurus theme, branding, and information architecture choices beyond
  basic nav structure — product/taste, decided by the maintainer separately.

## Assumptions (verified)

- crates.io `kremory` `max_version` = 0.7.0 — re-verified live via
  `https://crates.io/api/v1/crates/kremory` on 2026-09-07 (not carried over
  from memory).
- `docs/specs/` directory exists in this repo — confirmed by directory listing
  2026-09-07. **Corrected post-review (Finding 14)**: whether it was
  "pre-existing" vs "created moments earlier in this same session" cannot
  actually be established by a live directory listing (git does not track
  empty directories, so `git ls-tree` shows nothing either way). Not
  load-bearing for anything downstream — the qualifier is dropped rather than
  asserted past what the evidence supports.
- No `[NEEDS CLARIFICATION]` markers remain in this spec and no load-bearing
  assumption is unverified — `clarify` step is skippable per specflow's own
  footnote rule.

## Guardrails / scope constraints

- Do-not-touch: the 3 deliberately-held recall-scoring levers (see Non-goals).
- Do-not-touch: anything requiring a paid benchmark re-run, `cargo publish`,
  `git push`, or a release tag — all standing project HITL gates, unaffected
  by autonomous-mode execution of this spec.
- Required libraries/tooling: Docusaurus (Node/npm-based site generator) for
  Phase 4 — no alternative static-site generator substitution without a
  documented reason.

## Tier + gate rationale

**Tier: high-stakes** (elevated post-review from "non-trivial" — Finding 9).
specflow's own high-stakes trigger list is "release-critical, irreversible,
**phased**, or high-traffic" (any one qualifies), and this spec is explicitly
phased (Phase 0→5, each gating the next). The original "non-trivial"
classification under-called this criterion on its plain wording, and the
under-call had a concrete, demonstrated cost: a `risk-contingency-register`
would very plausibly have caught the napi-parity-cap collision (RULE-013)
before it became a review finding instead of a design input. Touches multiple
packages (crate, Node binding, new Docusaurus tooling), is substantially
AI-generated, and is shared (public docs, other consumers depend on behaviour
described here) — all independently true regardless of the phased trigger.

**`review` was mandatory regardless of tier** even before this elevation,
because Phase 0's correctness proof is compile-success against a golden set of
extracted code blocks — a test/golden-as-primary-evidence case, which the
specflow router flags as an override that beats the tier. It ran; see the
Review disposition section below.

**`risk-contingency-register` is now required** (high-stakes tier) — see
`risk-register.md` in this directory, added post-review.

**Spec-PR gate**: solo-maintainer project, no separate human review team.
Per specflow's own allowance for lighter/self-contained specs ("the author may
self-review and advance to Approved for Implementation"), the gate is
satisfied by the project owner's review in this conversation rather than a
formal GitHub PR round — there is no push in this spec's scope regardless
(standing HITL gate). Elevating to high-stakes changes which pre-gate
artifacts are mandatory (`review`, `risk-contingency-register`); it does not
reintroduce a formal multi-person PR round that doesn't exist for this
project.

## Review disposition (adversarial `specflow:review`, 2026-09-07)

Full report: see agent output referenced in this session. Verdict was
**"conditionally sound — 2 Blockers must resolve before implementation."**
Disposition of all 16 findings:

| # | Severity | Resolution |
|---|---|---|
| 1 | Blocker | Fixed — RULE-001 scaffold contract (per-section compilation + placeholder prelude) |
| 2 | Blocker | Fixed — RULE-012 (version-target labelling for unpublished fixes) |
| 3 | High | Fixed — `compile_fail` added as recognized marker (RULE-001) |
| 4 | High | Fixed — "compile ≠ run" restated explicitly in RULE-001 (was lost in the relocation from the superseded narrative doc) |
| 5 | High | Fixed — RULE-013 (napi-parity-cap guardrail) |
| 6 | Medium | Fixed — SC-010 |
| 7 | Medium | Fixed — RULE-006 disambiguated |
| 8 | Medium | Fixed — Non-goals disambiguated |
| 9 | Medium | Fixed — tier elevated to high-stakes, risk register added |
| 10 | Medium | Fixed — TD register's stale TD-231 status line corrected (`.ai-docs/tech-debt/tech-debt-register.md`) |
| 11 | Medium | Fixed — RULE-011/SC-006 strengthened to require execution/type-check |
| 12 | Low | Fixed — SC-011 |
| 13 | Low | Accepted as-is — a genuine live doc bug (`docs/api.md` default-features self-contradiction), correctly a Phase 1 finding rather than a spec defect. Pre-seeded in `plan.md` so Phase 1 doesn't have to rediscover it. |
| 14 | Low | Fixed — over-claimed "pre-existing" qualifier dropped from Assumptions |
| 15 | Low | Fixed — RULE-002 derivation stated inline |
| 16 | Low | Fixed — "Why now" 6-of-8 corrected to 4-of-8; uniform re-verify-at-kickoff instruction added |
