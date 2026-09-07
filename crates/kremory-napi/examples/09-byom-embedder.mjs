// kremory-napi `opts.withEmbedder` — BYOM (bring-your-own-model) embedder
// bridge (ADR-030 Tier-2). `Memory.open` with a custom embedder callback is
// exercised for real below. The ingest -> recall round-trip is NOT
// exercised here — it hits a pre-existing, already-tracked bridge bug.
//
// This mirrors (does not duplicate) `__test__/smoke-embedder.test.mjs`,
// which already covers: T1 (open succeeds), T3 (dim-mismatch error shape),
// T4 (backward-compat, no withEmbedder), T5 (null/undefined withEmbedder).
// That file's `test.skip('T2: ...')` documents the known-broken ingest
// round trip: "embedder callback error: InvalidArg, Given napi value is not
// an array ... the napi-rs ThreadsafeFunction returning Vec<f32> from an
// async JS callback hits a marshalling/lifetime issue in bridge.rs at
// ingest-time." This example RE-RUNS that exact scenario against the
// CURRENT build to confirm the bug is (or isn't) still live, rather than
// silently trusting a prior note.
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED (v0.4.1) — this imports the
// LOCALLY BUILT native module, not a published npm package.
//
// REQUIRES: native module built (`pnpm build:debug`).
//
// Run: `node examples/09-byom-embedder.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

const EMBED_DIM = 256;

function makeEmbedder(dim) {
  return async function embedText(text) {
    const safeText = text == null ? '' : String(text);
    const vec = new Array(dim).fill(0);
    for (let i = 0; i < safeText.length; i++) {
      vec[i % dim] += safeText.charCodeAt(i) / 255.0;
    }
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
    return vec.map((v) => v / norm);
  };
}

const dbPath = path.join(os.tmpdir(), `kremory-example-byom-${Date.now()}.db`);

console.log('[09] Memory.open() with opts.withEmbedder — the BYOM open path (ACTUALLY EXECUTED, real assertion)');
const mem = await Memory.open(dbPath, {
  withEmbedder: makeEmbedder(EMBED_DIM),
  embeddingDim: EMBED_DIM,
  defaultNamespace: 'byom-demo',
});
assert.ok(mem, 'Memory.open with withEmbedder must return a usable handle');
console.log('[09] open succeeded — handle is usable');

console.log('[09] remember() with the custom embedder — re-running the KNOWN, already-tracked bridge bug (see file header)');

// ── Finding F48 (logged in phase1-findings.md) ───────────────────────────
//
// `__test__/smoke-embedder.test.mjs`'s skipped T2 documents this call
// FAILING FAST with "InvalidArg, Given napi value is not an array".
// Re-running it against THIS build does something DIFFERENT and worse: it
// never resolves OR rejects — it HANGS. A 15s internal race (below) is used
// so this example can report that finding instead of hanging forever itself
// (a real consumer hitting this would have no such safety net). The bug has
// not been fixed; its FAILURE MODE has changed from a fast, descriptive
// error to a silent, indefinite hang — arguably a regression in observability
// even if the underlying defect is the same one.
const TIMEOUT_MS = 15_000;
let outcome;
try {
  outcome = await Promise.race([
    mem.remember({ content: 'BYOM round-trip re-check content.' }).then((r) => ({ kind: 'resolved', value: r })),
    new Promise((resolve) => setTimeout(() => resolve({ kind: 'timeout' }), TIMEOUT_MS)),
  ]);
} catch (err) {
  outcome = { kind: 'rejected', error: err };
}

if (outcome.kind === 'resolved') {
  console.log('[09] UNEXPECTED: remember() succeeded —', outcome.value);
  console.log('[09] the previously-tracked bug may be FIXED on this build. If so, update __test__/smoke-embedder.test.mjs to un-skip T2.');
} else if (outcome.kind === 'rejected') {
  console.log('[09] remember() failed (fast) as previously tracked:', outcome.error.message);
  assert.match(
    outcome.error.message,
    /InvalidArg|not an array|embedder/i,
    `expected the already-tracked embedder-marshalling error shape; got a DIFFERENT error: "${outcome.error.message}" — this may be a NEW, separate bug, worth its own finding`,
  );
} else {
  console.log(
    `[09] remember() neither resolved nor rejected within ${TIMEOUT_MS}ms — it HANGS on this build (Finding F48). ` +
      'This is a DIFFERENT failure mode than the fast "InvalidArg" error __test__/smoke-embedder.test.mjs documents for T2.',
  );
}

console.log('[09] PASS (example completed and reported ground truth; see Finding F48 if the hang occurred)');

// The BYOM embedder's ThreadsafeFunction can keep the event loop alive even
// after the call above times out (same class of napi-rs quirk as example
// 07). Force exit once the finding above has been reported.
process.exit(0);
