---
status: Draft
owner: "kgentic-dev"
jira: []
jira_note: "None — internal enhancement, surfaced in-session after the 0.8.0 crates.io publish"
spec_pr: null
implementation_pr: null
tier: 1-pr
---

# Implementation Spec: Use-case-shaped runnable examples, single-sourced to the docs site

## Links

- Design evidence base:
  - `.ai-docs/research/competitive-landscape/teardown-sqlite-graph-engines--source-level-20260908.md` — establishes which capabilities are differentiating (two-clock bi-temporal + active contradiction resolver; graph-from-text). 4:0 against the four engines examined.
  - `docs/api/bi-temporal.md` — the verified-compiling `as_of` surface this spec's flagship example demonstrates.
  - `docs/api/reversibility.md:115-125` — the verified-compiling `supersede` builder chain.
  - `website/scripts/sync-changelog.mjs` — the single-sourcing precedent this spec generalises.
  - `crates/kremory-doc-examples/README.md` — the existing doc-example compile gate.

## Summary

Give a new adopter runnable, use-case-shaped examples that demonstrate kremory's
differentiators, reachable from **both** the repo and the documentation site, with **one**
source of truth per example so the two cannot drift.

## Problem / context

kremory 0.8.0 went live on crates.io on 2026-09-08. A developer can now `cargo add kremory`.
⚠️ **Narrowed in revision 1** — the original claim ("nothing shows them how to use it") was
overstated. `docs/getting-started.md:80` has a section titled *"The smallest working example"*
with a full walkthrough, and `README.md:278-306` documents the two-clock temporal model in
prose. **The real gap is narrower and still real: nothing runs offline, and nothing is
organised around what the reader is trying to build.**

**What exists today, verified:**

- `crates/kremory/examples/` holds four files. Only `quickstart.rs` (92 lines) is user-facing,
  and its own module doc states it *"exists primarily as a **drift guard**"* for README
  snippets — it is a compile canary, not a teaching example. The other three
  (`alias_probe.rs` 269, `graph_health.rs` 305, `fastembed_rerank_spike.rs` 57) are internal
  diagnostics and spikes.
- **Examples do not ship.** `crates/kremory/Cargo.toml:23-31` is an `include = [...]`
  allowlist covering `src/**/*.rs`, `monitoring/provider-rates.toml`, `README.md`,
  `CHANGELOG.md`, `Cargo.toml`, `LICENSE`, `NOTICE`. Verified against the real packaged
  output: `target/package/kremory-0.8.0/` contains **no `examples/` directory**. crates.io
  and docs.rs users see zero examples.
- **The docs site has no use-case axis.** `website/sidebars.ts` is organised entirely by API
  surface — setup, namespaces, ingest, recall, bi-temporal, dream, reversibility,
  async-and-events, advanced, feature-flags, node-binding — explicitly in "adoption order".
  That answers *"how does `recall()` work"*. Nothing answers *"I want my agent to remember a
  user between sessions — where do I start?"*

The gap lands hardest on the differentiators. Every competitor can demonstrate
store-and-retrieve; per the teardown, none of the four engines examined has two-clock
bi-temporal or an active contradiction resolver. Those capabilities currently have **no
example at all**, so a reader cannot see what makes kremory different without assembling it
themselves from the API reference.

## Assumptions

- **ASM-001** — ⚠️ **CORRECTED (revision 1). As first written this was WRONG in the
  load-bearing way.** A `Memory` opens with an embedder and no LLM **only if a custom
  `EntityExtractor` is also wired**. `[verified EMPIRICALLY: running the example errors with
  "no extractor wired — call .with_llm(…) … or .with_extractor(Arc<impl EntityExtractor>)";
  statically at crates/kremory/src/facade/builder.rs:1458-1466]`.
  The original citation (`builder.rs:1428 — impl IntoFuture for MemoryBuilder<NoLlm, WithEmbedder>`)
  is real but proves only that the shape **compiles**; the runtime precondition sits 30 lines
  into the body. **A compile-level citation cannot verify a runtime precondition.** The
  corroborating file named its own missing precondition — `byoe-nollm-open.test.mjs`, where
  BYOE means *bring your own extractor*.
  Corroborated by `crates/kremory-napi/__test__/byoe-nollm-open.test.mjs`, a regression suite
  whose test 1 deliberately clears `OLLAMA_HOST`/`OPENAI_API_KEY`/`ANTHROPIC_API_KEY` to prove
  the NoLlm path. **This is the load-bearing assumption of the whole spec** — it is what makes
  an offline example possible.
- **ASM-002** — Facts can be written to the graph without LLM extraction, via
  `.with_facts(Vec<StructuredFact>)` combined with `.skip_extraction()`.
  `[verified: crates/kremory/src/facade/remember.rs:105 and :121 — both public methods on the remember builder]`.
- **ASM-003** — `MockChatProvider` is **not** usable by a consumer or a default-feature
  example. `[verified: crates/kremory/src/core/provider/chat.rs:93 — #[cfg(any(test, feature = "test-utils"))]]`.
  Therefore an offline example must reach ASM-001 + ASM-002, not a mock provider.
- **ASM-004** — Per-example feature gating is an established pattern in this crate.
  `[verified: crates/kremory/Cargo.toml:162-164 — [[example]] name = "fastembed_rerank_spike", required-features = ["rerank"]]`.
- **ASM-005** — A generated docs page derived from a repo file, gitignored and produced at
  `prebuild`, is an established pattern here.
  `[verified: website/scripts/sync-changelog.mjs + website/package.json:16-18 (sync:changelog wired into prebuild and prestart)]`.
  That script asserts its input shape and throws rather than emitting a plausible-but-wrong
  page — the behaviour this spec's generator copies.
- **ASM-006** — `cargo` builds examples during `cargo test`, so an example that stops
  compiling fails the existing gate without any new wiring.
  `[verified: crates/kremory/examples/quickstart.rs:3-6 — its module doc relies on exactly this property, and the crate has no [[example]] entry disabling it]`.
- **ASM-007** — `recorded_at` (transaction time) is returned on facts but is **not**
  server-side queryable; only valid-time is filterable via `.as_of()`.
  `[verified: docs/api/bi-temporal.md:70-73, a compile-gated page]`. The flagship example must
  therefore demonstrate the two clocks without implying a `recorded_at` filter exists.

## Intended behaviour

- **RULE-001** — WHERE an example demonstrates a capability that does not require entity
  extraction, the example SHALL open its `Memory` via the `NoLlm` type-state and SHALL NOT
  require any network service, external model, or non-default cargo feature to run.
- **RULE-002** — WHEN a reader runs `cargo run --example <name>` for any example designated
  offline under RULE-001, the example SHALL complete successfully with no environment
  variables set and no local service running.
- **RULE-003** — ⚠️ **REWRITTEN (revision 1).** The original required demonstrating
  `mem.supersede(fact_id)`. **That is unbuildable from the public surface**: `supersede` needs
  a `fact_id`, and neither thing a consumer can hold returns one — `remember()` returns
  `EpisodeCommit` (`memory/types.rs:837-863`, episode id only) and `recall().raw()` returns
  `RetrievedFact` (`memory/types.rs:439-465`, no id field). The crate's own e2e test obtains it
  from the graph layer (`tests/it/supersede_dream_closes_window.rs:87` —
  `graph.insert_fact_with_group(...)`), which consumers do not have. Filed as **TD-244**.
  **Revised rule:** The repository SHALL contain an example demonstrating, in one offline run:
  facts written with explicit validity windows, `recall` returning only the currently-true
  value, `.as_of(t)` with `t` inside the closed window returning the **superseded** value, and
  at least one fact observably present with a closed window (`RetrievedFact.invalid_at` is
  `Some`) rather than deleted.
- **RULE-004** — WHERE an example requires a live model because extraction is the capability
  being demonstrated, the example SHALL state that requirement in its module doc comment
  within the first 10 lines, naming the service and the model.
- **RULE-005** — WHEN the docs site is built, each published example page SHALL be generated
  from the corresponding `.rs` file, and the generated page SHALL NOT be committed to the
  repository.
- **RULE-006** — IF the example-sync generator cannot find its source file, or the source does
  not match the shape the generator expects, THEN the generator SHALL fail with a non-zero
  exit code and a message naming the offending file, and SHALL NOT emit a page.
- **RULE-007** — The example source SHALL NOT be duplicated into any committed markdown file
  by hand.
- **RULE-008** — WHEN `cargo test` runs, every example SHALL be compiled, such that an example
  that drifts from the public API fails the existing gate.
- **RULE-009** — Each published example SHALL be reachable from `website/sidebars.ts` under a
  use-case-named category distinct from the existing "API Reference" category.
- **RULE-010** — WHERE examples are included in the published crate, they SHALL be listed as
  **named files**, never as a `examples/**` glob, and `cargo package` SHALL succeed with them
  present. ⚠️ **Tightened in revision 1**: `crates/kremory/Cargo.toml:20-22` records that the
  `include` allowlist exists precisely to keep `examples/` out, because *"0.3.0 shipped ~181
  internal files for lack of this"*. A glob would republish `alias_probe.rs` and
  `graph_health.rs`, which are internal diagnostics.
- **RULE-011** — IF an offline example writes to the filesystem, THEN it SHALL write beneath a
  temporary or gitignored path and SHALL NOT leave artefacts in the working tree after a
  successful run.

## Non-goals

- **A tutorial series or cookbook section.** This spec adds examples and one navigation
  category, not a restructured information architecture.
- **Node/TypeScript example variants.** `@kgentic-ai/kremory-node` is unpublished and
  `kremory-napi` carries 92 parity gaps; examples in a language nobody can install are
  premature.
- **Rewriting or reorganising the existing API reference.** The API pages stay exactly as they
  are; this adds a parallel axis.
- **Demonstrating every capability.** Coverage is deliberately partial and moat-first.
- **Any `git push`, `cargo publish`, or release tag.**
- **Benchmark or eval runs.** No example is a performance claim.

## Scope constraints

- **Do not touch** `docs/api/**` content, `website/sidebars.ts` API Reference category
  membership, or the existing `quickstart.rs` drift-guard role.
- **Reuse, do not reinvent**, the `sync-changelog.mjs` shape for generation: assert the input,
  throw on surprise, gitignore the output, wire into `prebuild`.
- **Deterministic embedder pattern already exists** in `quickstart.rs` (`DemoEmbedder`, 16-dim)
  — reuse that approach rather than adding an embedding dependency.
- **Package size**: the published 0.8.0 crate is 4.7 MiB (1.2 MiB compressed). Example sources
  are plain text in the low kilobytes; any growth beyond ~1% of package size warrants a
  reconsideration recorded in the plan.

## Guardrails and constraints

- **No secrets, no PII.** Example fixture data SHALL be obviously synthetic. No real names,
  emails, or tokens.
- **API contract compatibility.** ⚠️ **Corrected in revision 1.** The original banned
  `kremory::core::*` as "internals". That premise is false: `crates/kremory/src/lib.rs:46`
  declares `pub mod core;`, so it is public, semver-covered API. The ban would also have made
  the offline path impossible, since `EntityExtractor` lives at `core::intelligence` and is not
  re-exported at the crate root. **Revised guardrail:** examples SHALL prefer root re-exports,
  and any `kremory::core::*` import SHALL carry an inline comment explaining why no root export
  exists. The missing re-export is filed as **TD-245**.
- **Offline-by-default is a security property, not just convenience.** An example that
  silently reaches the network on `cargo run` is prohibited by RULE-002.
- **Operational**: the docs build must remain deterministic and hermetic — the generator reads
  local files only and performs no network access.
- **No new runtime dependency** on the `kremory` crate. Examples may use existing dev/example
  dependencies only.

## Testing criteria

- **Unit** — the sync generator's input-assertion path: given a source file that does not
  match the expected shape, it exits non-zero and writes no output (RULE-006).
- **Integration** — `cargo test` compiles every example (RULE-008); `cargo run --example
  <offline-example>` completes with a cleared environment (RULE-002).
- **Behavioural** — the flagship example's own assertions verify RULE-003's five observations
  in-process, so the example is self-checking rather than merely printing.
- **Docs build** — `npm run build` in `website/` succeeds with the generated page present and
  reachable from the sidebar (RULE-005, RULE-009).
- **Packaging** — `cargo package --list` includes the example sources (RULE-010).
- **Regression** — the existing doc-example gate (`generate_docs.py` +
  `cargo test --doc -p kremory-doc-examples`) still passes at its current 26/26, and the
  workspace gate stays at its current 1800 passing.

## Success criteria

- **SC-001** — A developer with no Ollama, no API keys and no environment configuration can
  clone the repo and run the flagship example to completion with a single `cargo run
  --example` invocation.
- **SC-002** — The docs site presents at least one example page under a use-case category, and
  its content is byte-identical to the committed `.rs` source it derives from.
- **SC-003** — Deleting or renaming an example's source file causes the docs build to fail
  loudly rather than silently publish a stale or empty page.
- **SC-004** — `cargo package` output contains the example sources, so a crates.io or docs.rs
  visitor can read them without cloning.

## Spikes / validation needed

⚠️ **Revision 1 — the original text here said "None — all load-bearing assumptions verified".
That was the single worst line in the draft**: the one assumption it singled out as
load-bearing was the one that was wrong, and it was wrong in a way only *running* the code
could show. Recorded honestly:

- **SPIKE-001 — DONE, PASSED.** Does `NoLlm` + custom extractor + `.with_facts()` +
  `.skip_extraction()` + `recall()` work end to end offline? **Yes, verified by execution**
  (`cargo run --example offline_remember_recall`, exit 0, facts returned). There was no prior
  art in the tree — every existing `skip_extraction` test uses `MockChatProvider`, a
  `test-utils`-gated type consumers cannot reach — so this could only be settled by running it.
- **SPIKE-002 — DONE, PASSED.** Does `cargo package` succeed with a named example whose
  dependencies are dev-only (`tokio`, `anyhow`, `tempfile`)? **Yes**: 155 files, 4.8 MiB,
  exit 0. `--verify` compiles the library only and never builds example targets, so the
  dev-dependency concern does not materialise.

## Risks & mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | `.with_facts()` + `.skip_extraction()` may not populate the graph richly enough for `recall` to return the fact without an embedder match, making the flagship example's assertions fail for a reason unrelated to bi-temporality. | Medium | High — invalidates the flagship | Build the example assertion-first and run it before writing the docs half. If recall cannot see programmatically-written facts, fall back to asserting on `.raw()` facts directly and record the limitation in the example's own doc comment. |
| R2 | Including `examples/**` in the package makes `cargo package --verify` build examples against published-only dependencies, which may fail if an example uses a dev-dependency. | Medium | Medium | RULE-010 makes packaging an explicit gate. Verify with `cargo package` before committing the `include` change; if it fails, keep examples out of the package and record why. |
| R3 | A generated docs page that is gitignored is invisible in review, so a broken generator is only caught at build time. | Low | Medium | RULE-006 makes the generator fail loudly, and the docs build is part of the DoD below. |
| R4 | Adding a use-case category could imply broader coverage than one example delivers, reading as an unfinished section. | Medium | Low | Name the category for what it contains, and do not create placeholder entries for unwritten examples. |
| R5 | The deterministic embedder makes semantic recall trivially exact, so an example may imply better retrieval than a real embedder gives. | Low | Medium | The example's doc comment states that its embedder is a deterministic stand-in and links to the real embedder guidance. |

## Definition of Done

- [ ] RULE-001 / RULE-002 — the flagship example runs offline: verified by executing it with
      `env -i` (cleared environment) and no local service.
- [ ] RULE-003 — the flagship example asserts all five observations (recorded, superseded,
      recall-returns-new, as_of-returns-old, old-fact-bounded-not-deleted) in-process.
- [ ] RULE-004 — any model-requiring example names its service and model in the first 10 lines.
- [ ] RULE-005 / RULE-007 — the docs page is generated at `prebuild`, gitignored, and no
      example source is hand-copied into committed markdown (verified by grep).
- [ ] RULE-006 — generator failure path tested: a malformed/absent source exits non-zero and
      emits nothing.
- [ ] RULE-008 — `cargo test` compiles all examples.
- [ ] RULE-009 — the page is reachable from `website/sidebars.ts` under a non-API-Reference
      category.
- [ ] RULE-010 — `cargo package` succeeds and `cargo package --list` shows the example sources,
      OR the `include` change is reverted with the reason recorded (see R2).
- [ ] RULE-011 — no artefacts left in the working tree after a successful example run
      (`git status --porcelain` clean).
- [ ] SC-001 through SC-004 each demonstrated, not asserted.
- [ ] Gates: `/quality-gate` PASS · doc-example gate still 26/26 · workspace nextest still
      1800 passing · `npm run build` in `website/` exit 0.
- [ ] Guardrails: no example imports from `kremory::core::*`; fixture data obviously synthetic;
      no network access on any offline example.
