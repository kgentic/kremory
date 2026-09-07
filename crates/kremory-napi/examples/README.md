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

## Findings surfaced while writing these

Ten genuine issues were found by actually running new example content
against the locally built native module (none previously logged) — see
`docs/specs/public-docs-and-api-surface-audit/phase1-findings.md`, findings
**F40-F49**, appended by this same change. All are logged as **NOT FIXED**
— fixing any of them requires either a `crates/kremory/src` or
`crates/kremory-napi/src` change (out of this task's scope: writing and
running examples against the EXISTING build) or a maintainer product
decision. Summary (see the findings doc for full citations):

- **F40** — none of `supersede` / `deleteFact` / `restoreArchivedFact` /
  `unsupersede`'s required `factId: number` argument is discoverable from
  any public read path.
- **F41** — `awaitDream` / `cancelDream` require a dream-run `handleId`
  that no public JS (or Rust facade) call can produce, because
  `Memory.dream()` always blocks inline.
- **F42** — `recallBySourceId`'s "Known gap" doc comment (sourceUri always
  null) is STALE — sourceUri actually round-trips correctly (the fix it
  describes as missing already landed, TD-003 Phase G).
- **F43** — `awaitBatch` can NEVER report done for a batch of 2+ episodes
  sharing one `batchId` — always times out (a real substrate bug in
  `batch_status_increment_completed`/`_skipped`, which never bumps `total`
  past 1).
- **F44** — `restoreArchivedFact`'s documented idempotency
  (`alreadyLive: true`) is unreachable via the natural double-call
  sequence — the second call throws instead.
- **F45** — the BYOE `extractor.extract` callback's real calling
  convention (`(null, text)`, a leaked napi-rs error-first callback)
  matches NEITHER of the two shapes `index.d.ts` documents for it; `name`
  must be a plain string, not the documented `name(): string` method.
- **F46** — `{ embedder, extractor }`'s documented "NoLlm typestate" is not
  honoured — an LLM env var is still required to open a Memory this way.
- **F47** — BYOE-created entities fail to embed unless `embeddingDim`
  equals 384, regardless of the configured custom dimension.
- **F48** — the already-tracked BYOM-embedder ingest bug (`T2` in
  `smoke-embedder.test.mjs`) now HANGS instead of failing fast on this
  build — a failure-mode regression, not a new root cause.
- **F49** — `__test__/episode-parity.test.mjs` is stale and currently fails
  (`mem.getBySourceId is not a function`) against the real API surface;
  since it's in the default `pnpm test` glob, `pnpm test` does not pass
  cleanly on a fresh checkout today.

None of these were worked around silently — each is demonstrated and
commented on explicitly in its example file (see the Index table above), and
none required touching `src/` to write or run the examples themselves.
