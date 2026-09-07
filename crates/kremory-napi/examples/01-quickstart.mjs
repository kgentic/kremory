// kremory-napi quickstart — open → remember (real LLM extraction) → recall → close.
//
// Node-side mirror of `crates/kremory/examples/quickstart.rs`. Unlike the
// other examples in this directory, this one does NOT pass `skipExtraction`
// — it exercises the real Phase-2 LLM extraction path end-to-end, matching
// `__test__/smoke.test.mjs`'s "happy path" test.
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED (v0.4.1) — this imports the
// LOCALLY BUILT native module, not a published npm package.
//
// REQUIRES:
//   - `pnpm build:debug` already run in `crates/kremory-napi/` (native .node present)
//   - OLLAMA_HOST=http://localhost:11434 (or OPENAI_API_KEY / ANTHROPIC_API_KEY)
//     with `gemma4:e4b` + `nomic-embed-text` pulled if using Ollama
//
// Run: `node examples/01-quickstart.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

const dbPath = path.join(os.tmpdir(), `kremory-example-quickstart-${Date.now()}.db`);

console.log('[01] opening Memory at', dbPath);
const mem = await Memory.open(dbPath, { defaultNamespace: 'quickstart' });

try {
  console.log('[01] remember() — real LLM extraction (this can take 10-20s)...');
  const ingest = await mem.remember({
    content: 'James prefers concise, data-driven technical decisions and writes Rust.',
  });
  console.log('[01] ingest result:', ingest);
  assert.equal(typeof ingest.episodeEntityId, 'string', 'episodeEntityId must be a string');
  assert.ok(ingest.episodeEntityId.length > 0, 'episodeEntityId must be non-empty');
  assert.ok(!Number.isNaN(Date.parse(ingest.committedAt)), 'committedAt must be RFC-3339');
  // Inline path (no background enrichment requested) -> runId absent.
  assert.ok(ingest.runId == null, 'runId must be nullish for the default inline path');

  console.log('[01] recall("what does James prefer?")...');
  const results = await mem.recall('what does James prefer?', { k: 5 });
  assert.ok(Array.isArray(results), 'recall must return an array');
  console.log(`[01] recall returned ${results.length} result(s)`);
  for (const r of results) {
    console.log(`  - entityId=${r.entityId} name=${r.entityName} score=${r.score.toFixed(3)}`);
    assert.equal(typeof r.entityId, 'string');
    assert.equal(typeof r.entityName, 'string');
    assert.equal(typeof r.summary, 'string');
    assert.equal(typeof r.score, 'number');
    assert.ok(Array.isArray(r.sourceRefs));
    assert.ok(Array.isArray(r.facts));
  }

  console.log('[01] PASS');
} finally {
  await mem.close();
  try { fs.unlinkSync(dbPath); } catch { /* best-effort cleanup */ }
}
