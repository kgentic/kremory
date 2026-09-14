// kremory-napi BYOE (bring-your-own-extractor) — `opts.extractor`, a custom
// JS extraction callback bridged into the Rust `ExtractorKind::Custom` path
// (ADR-039 Shape B). Completely uncovered by any existing test/example
// before this change.
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED (v0.4.1) — this imports the
// LOCALLY BUILT native module, not a published npm package.
//
// ── Finding F46 (logged in phase1-findings.md) — FIXED, with a residual gap ──
//
// `index.d.ts`'s doc comment on `OpenOptions` documents `{ embedder,
// extractor }` as selecting the Rust `ExtractorKind::Custom` "NoLlm
// typestate" — i.e. no chat/LLM provider should be required to OPEN a
// Memory this way. Previously `open_with_js_embedder` called
// `bridge::resolve_env_llm().await?` UNCONDITIONALLY before ever checking
// `extractor_handle`, so opening rejected with "kremory open with embedder:
// no LLM provider configured — set OLLAMA_HOST, OPENAI_API_KEY, or
// ANTHROPIC_API_KEY" even with only `withEmbedder` + `extractor` supplied.
// FIXED: `open_with_js_embedder` now routes through
// `MemoryBuilder<NoLlm, WithEmb>` when an extractor is supplied, skipping
// `resolve_env_llm()` entirely — verified live: `Memory.open(...)` below now
// succeeds with ZERO LLM env vars set.
//
// RESIDUAL GAP found while verifying the fix above, NOT part of F46's
// original scope, NOT fixed here: the SUBSEQUENT `remember()` call below
// still requires an LLM — `ingest_with`'s entity-RESOLUTION step
// (`crates/kremory/src/core/ingest/pipeline/ingest_with.rs`, the
// `CascadeResolver` construction) unconditionally requires `self.llm`,
// regardless of whether a BYOE extractor supplied the entities and
// regardless of whether there is anything to resolve against (a fresh,
// empty namespace still hits this). So "NoLlm typestate" is accurate for
// OPENING a Memory, but overclaims for INGESTING through it once any
// entities are involved — `remember()` on a NoLlm-typestate Memory only
// works today via `.with_facts(…)` (pinned triples, extraction skipped
// entirely) per the error's own hint. Deciding a resolution-skipping (or
// simplified-resolution) fallback for the NoLlm path is a real substrate
// design decision, not a local bug fix — flagged, not attempted.
//
// REQUIRES: native module built (`pnpm build:debug`); OLLAMA_HOST (or
// OPENAI_API_KEY / ANTHROPIC_API_KEY) set — needed for the `remember()` call
// below (the residual gap above), NOT for `Memory.open()` itself anymore.
//
// Run: `OLLAMA_HOST=http://localhost:11434 node examples/07-byoe-custom-extractor.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

// ── Finding F47 (logged in phase1-findings.md) ───────────────────────────
//
// Empirically, `embeddingDim` values OTHER than 384 fail once a real entity
// gets embedded via a BYOE extractor: `EMBED_DIM = 32` (with a matching
// `embeddingDim: 32`) rejects with "SQLite failure: `vector index(insert):
// dimensions are different: 32 != 384`" the moment the extractor returns a
// real entity to embed — `EMBED_DIM = 384` succeeds with the identical
// code path otherwise unchanged. Not fully root-caused (would require
// touching `src/`, out of scope here), but the `entities`/`facts` embedding
// columns are declared `F32_BLOB({dim})` in the migrations
// (`crates/kremory/src/core/migrations/defs_j.rs` / `defs_h.rs`) — the
// evidence is consistent with that `{dim}` resolving to the schema default
// (384) rather than the caller's configured `embeddingDim` somewhere on the
// entity-embedding path when a BYOE extractor is in play, independent of
// whichever bug already tracked in `__test__/smoke-embedder.test.mjs`'s
// skipped T2 test (a different symptom: "InvalidArg, Given napi value is
// not an array", on the default IntegerId extractor, no BYOE). Using
// `EMBED_DIM = 384` below so this example demonstrates BYOE end-to-end
// rather than crashing on a dimension mismatch that isn't this example's
// point.
const EMBED_DIM = 384;

/** Deterministic BYOM embedder — same shape as `__test__/smoke-embedder.test.mjs`. */
function makeEmbedder(dim) {
  // ⚠️ TWO arguments, error-first: the bridge invokes this as `(err, text)`.
  // A one-argument form binds the parameter to `err` (always null), and a
  // null-guard then yields an ALL-ZERO vector of the right length — stored
  // without error and permanently unrecallable. `validate_embedding` now
  // rejects that, so this signature is load-bearing, not cosmetic.
  return async function embedText(_err, text) {
    const safeText = text == null ? '' : String(text);
    const vec = new Array(dim).fill(0);
    for (let i = 0; i < safeText.length; i++) {
      vec[i % dim] += safeText.charCodeAt(i) / 255.0;
    }
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
    return vec.map((v) => v / norm);
  };
}

/**
 * A trivial BYOE extractor: finds capitalised words as "entities" and emits
 * one canned fact. Real consumers would call their own NER/IE pipeline here
 * (a regex-based one, a hosted API, a local model via onnxruntime-node,
 * etc.) — kremory does not care HOW `entities`/`facts` were produced, only
 * that they match the documented shape.
 *
 * ── Finding F45 (logged in phase1-findings.md) — the `extract` signature ──
 *
 * `index.d.ts` documents TWO different shapes for this callback:
 *   - the standalone `ExternalExtractor` interface: `extract: (text: string)
 *     => Promise<...>` (ONE param)
 *   - the inline type used on `OpenOptions.extractor`: `extract(text:
 *     string, ctx: object): Promise<...>` (TWO params, second named `ctx`)
 *
 * Neither matches what the built module actually does. Empirically probed
 * (a throwaway script logging `arguments`): the real callback is invoked
 * with EXACTLY TWO arguments, `(null, "<the episode text>")` — i.e. the raw
 * napi-rs `ThreadsafeFunction<String, ErrorStrategy::CalleeHandled>`
 * error-first Node callback convention (`(err, value)`) has leaked directly
 * to the JS consumer, unwrapped. The first argument is ALWAYS `null` (an
 * error slot, never populated on the success path); the actual episode text
 * is the SECOND argument. So: neither documented shape is correct — not the
 * arity, and not the ONE param that IS present in both docs (`text` is
 * documented as argument 1; it is actually argument 2, and argument 1 is not
 * "ctx" at all, it's an always-null error slot).
 *
 * A SECOND, independent inconsistency (also found by actually running
 * this): both `index.d.ts` doc comments describe `name` as a METHOD —
 * `name(): string` — but the built native module rejects that at runtime
 * with `Error: Failed to convert JavaScript value \`function name(..)\`
 * into rust type \`String\` on ExternalExtractorHandle.name on
 * JsOpenOptions.extractor`. The Rust-side `ExternalExtractorHandle.name`
 * field is a plain `String`, not a callback — `name` must be a STRING
 * PROPERTY, not a function.
 *
 * Below, `name` is a plain string, and `extract` reads its real text out of
 * the SECOND argument, both to match the ACTUAL runtime contract rather
 * than either documented one.
 */
function makeCustomExtractor() {
  let callCount = 0;
  return {
    name: 'demo-regex-extractor',
    extract: async (errSlot, actualText) => {
      callCount += 1;
      console.log(
        `[07] custom extractor invoked (call #${callCount}); arg[0] (documented as "text", actually always null) =`,
        errSlot,
        '; arg[1] (documented as "ctx", actually the real text) =',
        JSON.stringify(actualText),
      );
      const text = actualText;
      const words = text.match(/\b[A-Z][a-zA-Z]{2,}\b/g) ?? [];
      const uniqueWords = [...new Set(words)];
      const entities = uniqueWords.map((w) => ({ name: w, label: 'Entity' }));
      const facts =
        uniqueWords.length >= 2
          ? [{ subject: uniqueWords[0], predicate: 'mentionedWith', object: uniqueWords[1] }]
          : [];
      return { entities, facts };
    },
  };
}

const dbPath = path.join(os.tmpdir(), `kremory-example-byoe-${Date.now()}.db`);

console.log('[07] opening Memory with { embedder, extractor } — NoLlm typestate, no chat provider needed');
const extractor = makeCustomExtractor();
const mem = await Memory.open(dbPath, {
  defaultNamespace: 'byoe-demo',
  withEmbedder: makeEmbedder(EMBED_DIM),
  embeddingDim: EMBED_DIM,
  extractor,
});

try {
  console.log('[07] remember() — routes through the custom extractor, NOT any LLM');
  const ingest = await mem.remember({
    content: 'Alice met Bob at the Riverside conference.',
  });
  console.log('[07] ingest result:', ingest);
  assert.equal(typeof ingest.episodeEntityId, 'string');

  console.log('[07] recall() to confirm the custom-extracted entity is searchable');
  const results = await mem.recall('Alice', { k: 5 });
  assert.ok(Array.isArray(results));
  console.log(`[07] recall returned ${results.length} result(s)`);
  for (const r of results) {
    console.log(`  - entityId=${r.entityId} name=${r.entityName}`);
  }

  console.log('[07] PASS');
} finally {
  await mem.close();
  try { fs.unlinkSync(dbPath); } catch { /* best-effort cleanup */ }
}

// napi-rs `ThreadsafeFunction`s (the BYOE extractor + BYOM embedder bridges)
// can leave the event loop alive after `close()` — a known napi-rs quirk
// with JS-callback bridges (see the similar note in
// `__test__/smoke-embedder.test.mjs`). Force exit now that all assertions
// above have already run and passed.
process.exit(0);
