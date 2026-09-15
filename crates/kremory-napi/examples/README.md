# kremory-napi examples

> ⚠️ **`@kgentic-ai/kremory-node` is UNPUBLISHED.** Current version `0.4.1`.
> It is **not** on npm — `npm install @kgentic-ai/kremory-node` does **not**
> work today. Every example below imports the **locally built** native module
> (`../index.js`, backed by the `kremory.darwin-arm64.node` / `.darwin-x64.node`
> binaries checked into this crate for local development) — there is no
> published package for a real external consumer to install yet. See the root
> `CLAUDE.md` "CURRENT STATE" table (`kremory-napi` row) for the up-to-date
> publish status.

These are **runnable** Node.js scripts, not narrative fragments — each one is
a standalone `.mjs` file you can execute directly with `node`, following the
same "walk it for real" discipline as `crates/kremory/examples/*.rs` (see
`quickstart.rs`, which this directory mirrors on the Node side) and the
existing `crates/kremory-napi/__test__/*.test.mjs` harness.

## Prerequisites

1. **Build the native module first** (if `../kremory.<platform>.node` isn't
   already present):
   ```sh
   cd crates/kremory-napi
   pnpm install && pnpm build:debug
   ```
2. **A local LLM provider for the examples that ingest with real extraction**
   (`01-quickstart.mjs`; everything else uses `skipExtraction: true` +
   pre-pinned `structuredFacts` so it runs fast and deterministically without
   needing a real extraction call):
   ```sh
   export OLLAMA_HOST=http://localhost:11434
   ollama pull gemma4:e4b
   ollama pull nomic-embed-text
   ```
   Every example that opens a `Memory` handle needs a chat provider configured
   at `Memory.open()` time even when the specific call it demonstrates passes
   `skipExtraction: true` — the provider is wired at open time, not per-call.
   `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` also work per the env-detection order
   documented on `Memory.open`.

## Running an example

```sh
cd crates/kremory-napi
export OLLAMA_HOST=http://localhost:11434
node examples/01-quickstart.mjs
```

Each script prints its own progress and exits non-zero on an unexpected
failure (`assert` from `node:assert/strict`), so `node examples/NN-*.mjs &&
echo OK` is a valid smoke check for any one of them.

## Index — capability → example

| Capability (per `../index.d.ts`) | Example | Actually executed? |
|---|---|---|
| `Memory.open` (Tier-1 env-auto), `remember`, `recall`, `close` — the happy path | `01-quickstart.mjs` | executed for real, real LLM extraction (gemma4:e4b via Ollama) |
| `defaultNamespace`, per-call `namespace`, `registerNamespace`, `upgradeNamespacePolicy` (AppendOnly), multi-namespace `recall` (`inNamespaces` + `bestEffort` + `perNamespaceTopK`) | `02-namespaces-and-policy.mjs` | executed for real (`skipExtraction: true`, deterministic) |
| `remember` full options (`sourceId`, `sourceUri`, `metadata`, `referenceTime`, `structuredFacts`, `skipExtraction`), `updateEpisodeMetadata`, `updateSourceUri`, `recallBySourceId`, `rememberBatch` + `awaitBatch`, `statusOf` | `03-ingest-batch-and-async-status.mjs` | executed for real (`skipExtraction: true`) |
| `recall` options: `k`, `asOf` (bi-temporal), `filterMetadata`, `rerankK`, `RetrievedContext.facts` / `RetrievedFact` shape | `04-recall-filters-and-bitemporal.mjs` | executed for real (`skipExtraction: true` + `structuredFacts` with `validFrom`/`validTo`) |
| `dream` (all `DreamOptions` knobs incl. `crossEpisodeMode`), `runDreamPassSync`, `ghostEpisodes`, `assertEntityType` | `05-dream-and-consolidation.mjs` | executed for real (`skipExtraction: true` for setup; `dream()`/`runDreamPassSync()` themselves are real, unmocked calls) |
| `supersede`, `unsupersede`, `mutationHistory`, `listMutations`, `undo` (unified dispatcher), `editEntity`/`undoEntityEdit`, `deleteEntity`/`undoDeleteEntity`, `deleteFact`/`undoDeleteFact`, `restoreArchivedFact` | `06-reversibility-and-undo.mjs` | executed for real — **see the file's top comment on the `factId` discoverability gap (Finding F40)** |
| BYOE — `opts.extractor` custom JS extraction callback (`ExternalExtractor` shape) | `07-byoe-custom-extractor.mjs` | executed for real |
| `opts.gliner` (GLiNER config) — feature-gated | `08-gliner-feature-gate.mjs` | executed for real, but the *documented rejection path*, not GLiNER extraction itself — the locally built `.node` was built with default features only (`content-search`; no `ner`), so this deliberately exercises `Memory.open`'s own documented `FeatureDisabled` error rather than a live GLiNER pass. Rebuilding with `--features ner` is out of scope for this example set (Rust-side build change). |
| `opts.withEmbedder` (BYOM embedder bridge) | `09-byom-embedder.mjs` | partial — `Memory.open` with a custom embedder is executed and asserted; the ingest-to-recall round trip is exercised too, but does NOT succeed — it hits a pre-existing, already-tracked bridge bug (`__test__/smoke-embedder.test.mjs` `test.skip('T2: ...')`, "InvalidArg, Given napi value is not an array"). Re-running it against this build surfaces a WORSE symptom than documented — see Finding F48 below. |
| `unmerge` (undo an entity merge) | `10-unmerge.ts` (type-check only) | NOT executed — see the file's header comment: triggering a real entity merge requires `dream()`'s cross-episode reconciliation to make an LLM/similarity-driven merge decision, which is not reliably reproducible as a deterministic example. Type-checked via `tsc` against `../index.d.ts` instead (mirrors `__test__/types.check.ts`'s own convention for shape-only verification). |

## Docs-site mirrors — narrative scenario ↔ Rust example

The table above covers the ten capability-smoke-test examples (`01`-`10`). A
second, smaller set exists for a different purpose: each is a faithful Node
mirror of one specific narrative example in `crates/kremory/examples/*.rs`,
kept in lockstep by `scripts/check-sdk-scenarios.sh` (runs both, diffs the
substance they print) and wired into the public docs site
(`website/scripts/sync-examples.mjs`'s `nodeFile` field) as a Rust/Node.js
language tab on that scenario's page. Add a new one here only once you have
a real Rust twin AND a `check-sdk-scenarios.sh` extractor that proves they
agree — a tab that isn't checked is a promise the repo can't keep.

| Node file | Mirrors | Docs page |
|---|---|---|
| `offline-remember-recall.mjs` | `offline_remember_recall.rs` | "Save something, get it back" |
| `remembers-across-sessions.mjs` | `remembers_across_sessions.rs` | "Remembering when things changed" |
| `multi-tenant-isolation.mjs` | `multi_tenant_isolation.rs` | "One database, many customers" |
| `undoing-a-correction.mjs` | `undoing_a_correction.rs` | "When the correction itself was wrong" |

## Automated smoke check

`pnpm test:examples` (`examples/run-all.mjs`) runs every `.mjs` file in this
directory (excluding itself) with a per-file timeout, and fails loud (exit 1,
listing which file(s) failed) if any exits non-zero or hangs past its
timeout. `10-unmerge.ts` is covered separately by `pnpm test:types`.

This exists **because** these examples had silently rotted before: F43 and
F44 below were fixed in the Rust/napi source on 2026-09-07, but the example
files that had been written to *demonstrate* those bugs were never updated
to match, and nothing ran them to notice — until this check was added and
re-running every example by hand caught it (2026-09-15). Requires the same
prerequisites as running any example manually (native module built, an LLM
provider env var set) — see "Prerequisites" above. There is no CI on this
crate (org-wide GitHub Actions billing is suspended), so **run this manually
before any napi release**, the same way `run-e2e-consumer.sh` is a manual,
mandatory pre-release gate on the Rust side.

## Findings surfaced while writing these (re-verified 2026-09-15)

Ten issues were originally found by actually running new example content
against the locally built native module — see
`docs/specs/public-docs-and-api-surface-audit/phase1-findings.md`, findings
**F40-F49**. That doc's own per-finding "Disposition" lines are the source
of truth and were already current; what was stale was THIS file and three
of the example files themselves, which still asserted the old bugs after the
fixes landed. Re-verified by actually re-running every example
(2026-09-15): **F43 and F44 no longer reproduce** — `examples/03-*.mjs` and
`examples/06-*.mjs` have been updated to assert the fixed (correct) behavior
instead of the historical bug. Current status of all ten, cross-checked
against the findings doc:

- **F40** — OPEN. None of `supersede` / `deleteFact` / `restoreArchivedFact` /
  `unsupersede`'s required `factId: number` argument is discoverable from
  any public read path. Still demonstrated in `06-reversibility-and-undo.mjs`.
- **F41** — Fixed (doc), 2026-09-07. `awaitDream`/`cancelDream` are documented
  as currently unreachable rather than claiming a handleId a caller can't get.
- **F42** — Fixed (doc), 2026-09-07. The stale "sourceUri always null" note
  was deleted (TD-003 Phase G already closed the gap it described).
- **F43** — **Fixed (code), 2026-09-07** (TD-251 follow-on work). `awaitBatch`
  now correctly reports done for a batch of 2+ episodes —
  `batch_status_increment_completed`/`_skipped` bump `total` on every arm,
  not just the first insert. `03-ingest-batch-and-async-status.mjs` now
  asserts the 3-episode batch completes, not that it times out.
- **F44** — **Fixed (code), 2026-09-07**. `restoreArchivedFact`'s
  idempotency (`alreadyLive: true` on a repeat call) is reachable again —
  `restore_archived_txn` now checks `facts` (already-live) before
  `facts_archive` presence. `06-reversibility-and-undo.mjs` now asserts the
  second call returns `alreadyLive: true`, not that it throws.
- **F45** — Fixed (doc), 2026-09-07. The BYOE `extractor.extract` callback
  shape in `index.d.ts` now matches the real napi-rs calling convention.
- **F46** — Fixed (code), 2026-09-07. `{ embedder, extractor }`'s "NoLlm
  typestate" is honoured when no LLM env var is present; `07-byoe-*.mjs` and
  `__test__/byoe-nollm-open.test.mjs` both pass this cleanly.
- **F47** — OPEN. BYOE-created entities still fail to embed unless
  `embeddingDim` equals 384, regardless of the configured custom dimension.
- **F48** — OPEN. The already-tracked BYOM-embedder ingest bug still hangs
  rather than failing fast. `09-byom-embedder.mjs` races the call against a
  15s timeout and reports ground truth either way (exits 0, self-documenting
  — this is intentional, not a gap in the example).
- **F49** — Fixed (test), 2026-09-07. `episode-parity.test.mjs` was renamed
  to the real `recallBySourceId` API; all 3 of its tests pass given a
  correctly configured LLM provider (the model actually pulled locally
  matters — `gemma4:e4b`, not just any Ollama model).

None of these were worked around silently — each open one is still
demonstrated and commented on explicitly in its example file, and none of
today's re-verification required touching `crates/kremory/src` or
`crates/kremory-napi/src`.
