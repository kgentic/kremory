// **Runs offline.** One database, many customers, and no leakage between them.
//
// Node mirror of `crates/kremory/examples/multi_tenant_isolation.rs`, scenario
// slug `multi-tenant-isolation`. The two are kept in lockstep by
// `scripts/check-sdk-scenarios.sh`, which runs BOTH and asserts they agree — so
// the docs-site language tabs cannot show a Node snippet that no longer works.
//
// ## The problem this solves
//
// You are building an assistant that serves several customers from one process.
// Acme's facts must never surface in a Globex answer. Getting that wrong is not
// a bug report, it is a disclosure incident.
//
// kremory's unit of isolation is the **namespace**. It is not a filter applied
// after the fact — it scopes the query itself, so an omitted namespace cannot
// silently widen a result set.
//
// ## What this example asserts, not just prints
//
// It writes facts for two tenants into ONE database and then proves, with hard
// assertions rather than eyeballing, that:
//
//   - each tenant sees its own facts,
//   - neither tenant sees the other's,
//   - the isolation holds for a term that appears in BOTH tenants' data
//     (the interesting case — a shared word is where a naive filter leaks).
//
// If a future change breaks scoping, this example stops exiting 0.
//
// No Ollama. No API keys. No environment variables. No network.
//
// ⚠️ Opening offline needs BOTH a custom embedder AND a custom extractor.
// Supplying only `withEmbedder` takes a different branch that hard-requires an
// env LLM (see `offline-remember-recall.mjs` for the full explanation).
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED (v0.4.1) — this imports the
// LOCALLY BUILT native module, not a published npm package.
//
// REQUIRES: native module built (`pnpm build:debug` in crates/kremory-napi/).
//
// Run: `node examples/multi-tenant-isolation.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

const DEMO_DIM = 16;

/**
 * A deterministic stand-in so this example needs no embedding service.
 * **Not for production** — it hashes bytes, it does not understand meaning.
 *
 * Byte-for-byte the same arithmetic as `DemoEmbedder` in the Rust example, so
 * both SDKs produce identical vectors for identical text.
 */
function demoEmbedder(dim) {
  // ⚠️ TWO arguments — error-first ThreadsafeFunction convention. `err` is
  // always null and the TEXT IS THE SECOND ARGUMENT. See
  // `offline-remember-recall.mjs` for why a one-argument callback silently
  // corrupts every embedding instead of erroring.
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
 * Never invoked — every write here calls `skipExtraction: true`. See
 * `offline-remember-recall.mjs` for why the builder still requires one.
 *
 * NB `extract` takes napi-rs's error-first arguments: `err` is always null
 * and the text is the SECOND argument.
 */
const noExtraction = {
  name: 'no-extraction',
  extract: async (_err, _text) => ({ entities: [], facts: [] }),
};

/** Everything one tenant said, and the answer that must never cross the boundary. */
const TENANTS = [
  {
    namespace: 'acme',
    subject: 'acme',
    // Deliberately the same PREDICATE for both tenants — a shared term is
    // exactly where a filter-after-the-fact implementation leaks.
    secretTool: 'Postgres',
  },
  {
    namespace: 'globex',
    subject: 'globex',
    secretTool: 'ClickHouse',
  },
];

const dbPath = path.join(os.tmpdir(), `kremory-example-multi-tenant-${Date.now()}.db`);

const mem = await Memory.open(dbPath, {
  embeddingDim: DEMO_DIM,
  defaultNamespace: 'unused-default',
  withEmbedder: demoEmbedder(DEMO_DIM),
  extractor: noExtraction,
});

try {
  // ── Write both tenants into the SAME database ───────────────────────────
  for (const t of TENANTS) {
    await mem.remember({
      content: `Internal notes for ${t.subject}.`,
      namespace: t.namespace,
      structuredFacts: [
        // Same predicate on both sides — the shared-term case.
        { subject: t.subject, predicate: 'runs_on', object: t.secretTool },
      ],
      skipExtraction: true,
    });
  }

  // ── Prove isolation, per tenant ─────────────────────────────────────────
  for (const t of TENANTS) {
    const contexts = await mem.recall('what does the company run on', {
      namespace: t.namespace,
    });
    const objects = contexts.flatMap((c) => c.facts).map((f) => f.object);

    console.log(`${t.subject.padStart(7)} sees: ${JSON.stringify(objects)}`);

    // Its own fact is present …
    assert.ok(
      objects.some((o) => o === t.secretTool),
      `${t.subject} should see its own fact ${JSON.stringify(t.secretTool)}, got ${JSON.stringify(objects)}`,
    );

    // … and no other tenant's is.
    for (const other of TENANTS) {
      if (other.subject === t.subject) continue;
      assert.ok(
        !objects.some((o) => o === other.secretTool),
        `LEAK: ${t.subject} saw ${other.subject}'s fact ${JSON.stringify(other.secretTool)} — got ${JSON.stringify(objects)}`,
      );
    }
  }

  console.log('\nBoth tenants live in one file and neither can see the other.');
  console.log('The namespace scopes the QUERY — it is not a filter applied afterwards,');
  console.log('so forgetting to pass one cannot silently widen a result set.');
} finally {
  await mem.close();
  try { fs.unlinkSync(dbPath); } catch { /* best-effort cleanup */ }
}

// A custom `withEmbedder` callback (this example uses one) leaves a napi-rs
// ThreadsafeFunction handle that keeps the event loop alive past process
// completion — the same documented bridge quirk `offline-remember-recall.mjs`
// and `09-byom-embedder.mjs` already force-exit for. Force exit once
// everything above has genuinely completed.
process.exit(0);
