// Node mirror of `crates/kremory/examples/remembers_across_sessions.rs`, scenario
// slug `remembers-across-sessions`.
//
// No Ollama. No API keys. No environment variables. No network.
//
// ## What this shows
//
// An agent learns something about a user, the world changes, and the agent has to
// answer BOTH questions correctly:
//
//   - "Where does Alice live?"                  -> Berlin  (what is true now)
//   - "Where did Alice live back before March?"  -> London  (what was true then)
//
// Most memory systems can only answer the first, because an update overwrites the
// old value. kremory keeps both, because a fact carries a *validity window* — so the
// old answer is still there, correctly bounded, rather than deleted.
//
// ## The two clocks (this is the part worth understanding)
//
// Every fact has two independent timelines:
//
//   - **valid time** — when the fact was true *in the world*. Mutable: London was
//     true until March, then stopped being true. Queried with `asOf` on `recall()`.
//   - **transaction time** (`recordedAt`) — when the database *learned* it.
//     Immutable, and it never changes even when the fact is later corrected.
//
// They come apart constantly in real systems: you can learn in June that someone
// moved in March. Valid time says March; transaction time says June. Collapsing them
// into one "updatedAt" column loses the distinction permanently.
//
// ## A naming trap worth knowing before you read the fields
//
// `recall()` gives you `RetrievedFact` objects on `context.facts`, whose fields are
// named relative to the world clock, not the stored row:
//
// | `RetrievedFact` field | means                          |
// |------------------------|---------------------------------|
// | `validAt`              | when it became true            |
// | `invalidAt`             | when it stopped being true (if ever) |
// | `recordedAt`           | when we learned it             |
//
// So `invalidAt` present = the validity window has CLOSED — the fact is superseded,
// not deleted.
//
// ⚠️ Opening offline needs BOTH a custom embedder AND a custom extractor.
// Supplying only `withEmbedder` takes a different branch that hard-requires an
// env LLM — see `offline-remember-recall.mjs` for the detail. Supplying an
// extractor as well (as this example does) selects the no-LLM path.
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED — this imports the LOCALLY BUILT
// native module, not a published npm package.
//
// REQUIRES: native module built (`pnpm build:debug` in crates/kremory-napi/).
//
// Run: `node examples/remembers-across-sessions.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

/** Dimension of the stand-in embedder below. Must be declared via
 * `embeddingDim` because the default (384) only applies to the built-in
 * embedder path. */
const DEMO_DIM = 16;
const NAMESPACE = 'alice-assistant';

/**
 * A deterministic, non-semantic stand-in so this example needs no embedding
 * service. **Not for production** — it hashes bytes, it does not understand
 * meaning. Byte-for-byte the same arithmetic as `DemoEmbedder` in the Rust
 * example, so both SDKs produce identical vectors for identical text.
 *
 * ⚠️ TWO arguments, error-first (napi-rs ThreadsafeFunction convention):
 * `err` is always `null` and the TEXT IS THE SECOND ARGUMENT. See
 * `offline-remember-recall.mjs` for why a one-argument callback silently
 * produces an all-zero, permanently-unrecallable vector.
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
 * Never invoked — the facts below are supplied directly via
 * `structuredFacts` and `skipExtraction: true` turns Phase-2 extraction off.
 * It exists because opening without an LLM requires saying HOW facts would
 * be extracted, exactly as the Rust example's `NoExtraction` does.
 *
 * NB `extract` takes napi-rs's error-first arguments: `err` is always `null`
 * and the text is the SECOND argument.
 */
const noExtraction = {
  name: 'no-extraction',
  extract: async (_err, _text) => ({ entities: [], facts: [] }),
};

const dbPath = path.join(
  os.tmpdir(),
  `kremory-example-remembers-across-sessions-${Date.now()}.db`,
);

const DAY_MS = 24 * 60 * 60 * 1000;
const now = new Date();
const march = new Date(now.getTime() - 120 * DAY_MS);
const beforeMove = new Date(march.getTime() - 30 * DAY_MS);
const londonMoveIn = new Date(march.getTime() - 365 * DAY_MS);

// Open with an embedder and NO LLM. An LLM is only needed when kremory has to
// EXTRACT facts from prose. Here we supply the facts ourselves, so there is
// nothing to extract and nothing to call.
const mem = await Memory.open(dbPath, {
  embeddingDim: DEMO_DIM,
  defaultNamespace: NAMESPACE,
  withEmbedder: demoEmbedder(DEMO_DIM),
  extractor: noExtraction,
});

try {
  // ── Record Alice's location history ─────────────────────────────────────
  //
  // `structuredFacts` + `skipExtraction: true` writes facts directly — no LLM
  // in the loop. This is also how you'd import an existing dataset.
  //
  // Each fact carries its own validity WINDOW. London is bounded (it stopped
  // being true in March); Berlin is open-ended (still true). Both rows live
  // in the graph.
  await mem.remember({
    content: 'Alice lived in London, then moved to Berlin in March.',
    namespace: NAMESPACE,
    structuredFacts: [
      {
        subject: 'alice',
        predicate: 'lives_in',
        object: 'London',
        validFrom: londonMoveIn.toISOString(),
        validTo: march.toISOString(), // the window CLOSES — not a delete
      },
      {
        subject: 'alice',
        predicate: 'lives_in',
        object: 'Berlin',
        validFrom: march.toISOString(), // still true — no validTo
      },
    ],
    skipExtraction: true,
  });

  // ── "Where does Alice live?" — as of NOW ────────────────────────────────
  const nowContexts = await mem.recall('where does alice live', { namespace: NAMESPACE });
  const nowFacts = nowContexts.flatMap((c) => c.facts);
  const nowObjects = nowFacts.map((f) => f.object);

  console.log(`now      -> ${JSON.stringify(nowObjects)}`);
  assert.ok(
    nowObjects.some((o) => o.toLowerCase() === 'berlin'),
    `expected Berlin in a present-time recall, got ${JSON.stringify(nowObjects)}`,
  );

  // ── "Where did Alice live before March?" — as of a PAST date ────────────
  //
  // `asOf` is a VALID-TIME query: what was true in the world at that instant.
  // Ask about a date before the move and London is the answer that comes
  // back instead of Berlin.
  const thenContexts = await mem.recall('where does alice live', {
    namespace: NAMESPACE,
    asOf: beforeMove.toISOString(),
  });
  const thenFacts = thenContexts.flatMap((c) => c.facts);
  const thenObjects = thenFacts.map((f) => f.object);

  console.log(`as_of(before move) -> ${JSON.stringify(thenObjects)}`);

  // ── The old fact is BOUNDED, not deleted ────────────────────────────────
  //
  // This is the property that distinguishes a bi-temporal store from an
  // overwrite: the superseded value is still on disk with a closed window,
  // and its `recordedAt` still says when we first learned it.
  const closedWindowCount = [...nowFacts, ...thenFacts].filter(
    (f) => f.invalidAt != null,
  ).length;

  // Only `nowFacts` — the same closed fact also comes back from the asOf
  // query, and printing it twice reads as a bug rather than as two views of
  // one row.
  for (const f of nowFacts.filter((f) => f.invalidAt != null)) {
    console.log(
      `bounded  -> ${f.subject} ${f.predicate} valid[${f.validAt.slice(0, 10)} .. ${f.invalidAt.slice(0, 10)}]  recordedAt=${f.recordedAt.slice(0, 10)}`,
    );
  }

  assert.ok(
    thenObjects.some((o) => o.toLowerCase() === 'london'),
    `expected London when asking about a date before the move, got ${JSON.stringify(thenObjects)}`,
  );
  assert.ok(
    closedWindowCount > 0,
    'expected at least one fact with a CLOSED validity window — ' +
      'the whole point is that superseded facts are bounded, not deleted',
  );

  console.log('\nBoth answers are correct at once, because a fact has a WINDOW, not');
  console.log('a single timestamp. Nothing was overwritten and nothing was deleted.');
} finally {
  await mem.close();
  try { fs.unlinkSync(dbPath); } catch { /* best-effort cleanup */ }
}

// A custom `withEmbedder` callback (this example uses one) leaves a napi-rs
// ThreadsafeFunction handle that keeps the event loop alive past process
// completion — the same documented bridge quirk `offline-remember-recall.mjs`
// already force-exits for. Force exit once everything above has genuinely
// completed.
process.exit(0);
