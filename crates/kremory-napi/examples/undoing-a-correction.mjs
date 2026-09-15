// **Runs offline.** You corrected a record. The correction was wrong. Put it
// back.
//
// Node mirror of `crates/kremory/examples/undoing_a_correction.rs`, scenario
// slug `undoing-a-correction`. Kept in lockstep with the Rust example by
// `scripts/check-sdk-scenarios.sh`, which runs BOTH and asserts they agree —
// so the docs-site language tabs cannot show a Node snippet that no longer
// works.
//
// ## The problem this solves
//
// `correcting-the-record` closes a fact on the world clock — "she worked
// there until March". Then someone checks and it turns out she never left.
// The correction was the error.
//
// `unsupersede()` clears the bound and the fact is current again. It takes a
// `factId`, NOT a mutationId, so it pairs directly with what `recall()`
// already hands you (modulo the F40 caveat below).
//
// ## An outcome worth copying
//
// `UnsupersedeOutcome.outcome` is `"cleared"` or `"not_superseded"` — not a
// boolean. Calling it on a fact that was never superseded returns
// `"not_superseded"` rather than a success that means nothing. Both branches
// are exercised below, because the second is the one that tells you your
// assumption was wrong.
//
// ── Finding F40 (see docs/specs/public-docs-and-api-surface-audit/phase1-findings.md,
// and `06-reversibility-and-undo.mjs`'s header comment for the full writeup) ──
//
// `supersede` / `unsupersede` take a numeric `factId` as their primary
// argument, and there is NO public path to discover a fact's numeric id from
// `recall()` — `RetrievedFact` carries no `id` field on either the Rust
// facade or this binding. This example relies on the same
// deterministic-but-unsupported implementation detail as `06`: in a FRESH
// database, inserting exactly one `structuredFacts` entry via `remember()`
// gives fact id `1`. **This is NOT a documented or supported pattern** — it
// is used here only to exercise `supersede`/`unsupersede` deterministically.
//
// No Ollama. No API keys. No environment variables. No network.
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED (v0.4.1) — this imports the
// LOCALLY BUILT native module, not a published npm package.
//
// REQUIRES: native module built (`pnpm build:debug` in crates/kremory-napi/).
//
// Run: `node examples/undoing-a-correction.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

const DEMO_DIM = 16;
const NS = 'hr';

/**
 * A deterministic stand-in so this example needs no embedding service.
 * Byte-for-byte the same arithmetic as `DemoEmbedder` in the Rust example
 * (and identical to the one in `offline-remember-recall.mjs`), so both SDKs
 * produce identical vectors for identical text.
 *
 * ⚠️ TWO arguments — napi-rs ThreadsafeFunction error-first convention:
 * `(err, text) => ...`, not `(text) => ...`. `err` is always null here; the
 * text is the SECOND argument.
 */
function demoEmbedder(dim) {
  return async function embed(_err, text) {
    const safeText = text == null ? '' : String(text);
    const vec = new Array(dim).fill(0);
    for (let i = 0; i < safeText.length; i++) {
      vec[i % dim] += safeText.charCodeAt(i) / 255.0;
    }
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1e-6;
    return vec.map((v) => v / norm);
  };
}

/**
 * Never invoked — the fact below is supplied directly and `skipExtraction`
 * turns Phase-2 off. Opening offline still requires saying HOW facts would
 * be extracted, exactly as the Rust example's `NoExtraction` does.
 *
 * NB `extract` takes napi-rs's error-first arguments: `err` is always null
 * and the text is the SECOND argument.
 */
const noExtraction = {
  name: 'no-extraction',
  extract: async (_err, _text) => ({ entities: [], facts: [] }),
};

const dbPath = path.join(os.tmpdir(), `kremory-example-undoing-a-correction-${Date.now()}.db`);

console.log('[undoing-a-correction] opening Memory (fresh db — fact id below is deterministic, see F40 header comment)');
const mem = await Memory.open(dbPath, {
  embeddingDim: DEMO_DIM,
  defaultNamespace: NS,
  withEmbedder: demoEmbedder(DEMO_DIM),
  extractor: noExtraction,
});

/** Mirrors the Rust example's `current()` helper. */
async function current(memory) {
  const contexts = await memory.recall('wren', { namespace: NS });
  return contexts
    .flatMap((c) => c.facts)
    .filter((f) => f.predicate === 'works_at')
    .map((f) => f.object);
}

try {
  await mem.remember({
    content: 'Wren works at Halden Institute.',
    namespace: NS,
    structuredFacts: [{ subject: 'wren', predicate: 'works_at', object: 'Halden Institute' }],
    skipExtraction: true,
  });

  // Fresh db, single structuredFacts insert above → fact id 1. See the F40
  // header comment; this is a deliberate, called-out reliance on an
  // undocumented implementation detail, not a supported API.
  const factId = 1;
  console.log('recorded            :', await current(mem));

  // ── The correction, which will turn out to be wrong ─────────────────────
  //
  // `validTo: now` + `closeNow: true` closes the fact right now — by the
  // time the inline close-out sweep runs, `now` is already in the past. See
  // `06-reversibility-and-undo.mjs` for why a future-dated bound alone
  // leaves the fact still reading as current (`retired` stays `0`); a
  // validTo BEFORE the fact's own `validFrom` is instead rejected as a time
  // inversion, which is why this uses "now", not a backdated timestamp.
  const correctionAt = new Date().toISOString();
  console.log(`[undoing-a-correction] supersede(${factId}, ${correctionAt}, "heard she left", closeNow: true)`);
  const supersedeResult = await mem.supersede(factId, correctionAt, 'heard she left', NS, true);
  console.log('[undoing-a-correction] supersede result:', supersedeResult);
  assert.equal(supersedeResult.outcome, 'bounded');

  const after = await current(mem);
  console.log("after 'correction'  :", after);
  assert.equal(after.length, 0, `the supersession should have closed the fact; got ${JSON.stringify(after)}`);

  // ── It was wrong. She never left. ───────────────────────────────────────
  const outcome = await mem.unsupersede(factId);
  console.log('unsupersede         :', outcome);
  assert.equal(outcome.outcome, 'cleared');
  assert.equal(outcome.factId, factId);
  assert.equal(outcome.clearedValidTo, true);

  const restored = await current(mem);
  console.log('after undo          :', restored);
  assert.ok(
    restored.some((o) => o === 'Halden Institute'),
    `clearing the bound should make the fact current again; got ${JSON.stringify(restored)}`,
  );

  // ── And the honest no-op ────────────────────────────────────────────────
  //
  // Calling it again: the bound is already cleared, so there is nothing to
  // do and the outcome SAYS SO rather than returning a meaningless success.
  const again = await mem.unsupersede(factId);
  console.log('unsupersede again   :', again);
  assert.equal(
    again.outcome,
    'not_superseded',
    `a second call should report "not_superseded", not a success that means nothing; got ${JSON.stringify(again)}`,
  );

  console.log("\nFour tools, four situations — and this is the one for 'the correction");
  console.log('itself was the mistake\'. Nothing was ever deleted, so it was always');
  console.log('recoverable: a closed fact is closed, not gone.');

  console.log('[undoing-a-correction] PASS');
} finally {
  await mem.close();
  try { fs.unlinkSync(dbPath); } catch { /* best-effort cleanup */ }
}

// A custom `withEmbedder` callback (this example uses one) leaves a napi-rs
// ThreadsafeFunction handle that keeps the event loop alive past process
// completion — the same known bridge quirk `offline-remember-recall.mjs` and
// `09-byom-embedder.mjs` already force-exit for. Force exit once everything
// above has genuinely completed.
process.exit(0);
