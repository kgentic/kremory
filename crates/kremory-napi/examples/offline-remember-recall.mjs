// **Start here.** Save something, get it back — offline, in about thirty seconds.
//
// Node mirror of `crates/kremory/examples/offline_remember_recall.rs`, scenario
// slug `offline-remember-recall`. The two are kept in lockstep by
// `scripts/check-sdk-scenarios.sh`, which runs BOTH and asserts they agree — so
// the docs-site language tabs cannot show a Node snippet that no longer works.
//
// No Ollama. No API keys. No environment variables. No network.
//
// ⚠️ Opening offline needs BOTH a custom embedder AND a custom extractor.
// Supplying only `withEmbedder` takes a different branch that hard-requires an
// env LLM (`crates/kremory-napi/src/lib.rs`, the `resolve_env_llm()` call after
// the `extractor_handle` block) — which is why `09-byom-embedder.mjs` cannot run
// without one. Supplying an extractor as well selects the no-LLM path.
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED (v0.4.1) — this imports the
// LOCALLY BUILT native module, not a published npm package.
//
// REQUIRES: native module built (`pnpm build:debug` in crates/kremory-napi/).
//
// Run: `node examples/offline-remember-recall.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

const DEMO_DIM = 16;
const NAMESPACE = 'demo';

/**
 * A deterministic stand-in so this example needs no embedding service.
 * **Not for production** — it hashes bytes, it does not understand meaning.
 * A real consumer calls their embedding backend here (OpenAI, Ollama
 * `nomic-embed-text`, a local GGUF, sentence-transformers over HTTP).
 *
 * Byte-for-byte the same arithmetic as `DemoEmbedder` in the Rust example, so
 * both SDKs produce identical vectors for identical text.
 */
function demoEmbedder(dim) {
  // ⚠️ TWO arguments. The embedder is invoked through a napi-rs
  // ThreadsafeFunction with the error-first calling convention, exactly like
  // `opts.extractor.extract` — `err` is always null and the TEXT IS THE SECOND
  // ARGUMENT. A one-argument `async (text) => …` silently binds `text` to null,
  // and a null-guard that maps it to '' then returns an ALL-ZERO vector of the
  // right length. It passes the bridge's dimension check, stores cleanly, and
  // has zero cosine similarity with every query — so entities embed to nothing
  // and become permanently unrecallable, with no error anywhere.
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
 * Never invoked — the facts below are supplied directly and `skipExtraction`
 * turns Phase-2 off. It exists because opening without an LLM requires saying
 * HOW facts would be extracted, exactly as the Rust example's `NoExtraction`
 * does.
 *
 * NB `extract` takes napi-rs's error-first arguments: `err` is always null and
 * the text is the SECOND argument.
 */
const noExtraction = {
  name: 'no-extraction',
  extract: async (_err, _text) => ({ entities: [], facts: [] }),
};

const dbPath = path.join(os.tmpdir(), `kremory-example-offline-${Date.now()}.db`);

const mem = await Memory.open(dbPath, {
  embeddingDim: DEMO_DIM,
  defaultNamespace: NAMESPACE,
  withEmbedder: demoEmbedder(DEMO_DIM),
  extractor: noExtraction,
});

try {
  // Save two things we know about a user.
  await mem.remember({
    content: 'Notes about Jim.',
    namespace: NAMESPACE,
    structuredFacts: [
      { subject: 'jim', predicate: 'writes', object: 'Rust' },
      { subject: 'jim', predicate: 'prefers', object: 'concise replies' },
    ],
    skipExtraction: true,
  });

  // Get them back. `recall()` gives you the structured facts;
  // `recallAsPromptText()` gives you a rendered string ready to drop into a
  // prompt instead.
  const contexts = await mem.recall('what do we know about jim', { namespace: NAMESPACE });
  const facts = contexts.flatMap((c) => c.facts);

  for (const f of facts) {
    console.log(`${f.subject} ${f.predicate} ${f.object}`);
  }

  assert.ok(
    facts.length > 0,
    'expected to recall the facts we just stored. If this fails with KREMORY_* ' +
      'environment variables set, unset them — they override recall scoring.',
  );

  const rendered = await mem.recallAsPromptText('what do we know about jim', null, {
    namespace: NAMESPACE,
  });
  console.log(`\n--- prompt-ready ---\n${rendered}`);
} finally {
  await mem.close();
  try { fs.unlinkSync(dbPath); } catch { /* best-effort cleanup */ }
}

// A custom `withEmbedder` callback (this example uses one, per the module
// doc above) leaves a napi-rs ThreadsafeFunction handle that keeps the
// event loop alive past process completion — the same known bridge quirk
// `09-byom-embedder.mjs` already force-exits for. Discovered here via
// `scripts/check-sdk-scenarios.sh` (the process printed correct output,
// including the closing "prompt-ready" block, then never exited on its
// own). Force exit once everything above has genuinely completed.
process.exit(0);
