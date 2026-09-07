// kremory-napi recall() options — k, asOf (bi-temporal filtering),
// filterMetadata, rerankK, and the RetrievedContext/RetrievedFact shape
// (connected facts returned inline on each recall row per ADR-074/TD-116).
//
// Uses `skipExtraction: true` + `structuredFacts` throughout so recall
// results are deterministic (no dependency on real LLM extraction quality).
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED (v0.4.1) — this imports the
// LOCALLY BUILT native module, not a published npm package.
//
// REQUIRES: native module built (`pnpm build:debug`);
//   OLLAMA_HOST=http://localhost:11434 (or OPENAI_API_KEY / ANTHROPIC_API_KEY).
//
// Run: `node examples/04-recall-filters-and-bitemporal.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

const dbPath = path.join(os.tmpdir(), `kremory-example-recall-filters-${Date.now()}.db`);

console.log('[04] opening Memory');
const mem = await Memory.open(dbPath, { defaultNamespace: 'recall-demo' });

try {
  // ── Seed data: a fact valid only in a specific historical window ────────
  console.log('[04] remember() with structuredFacts + a bounded validFrom/validTo window');
  await mem.remember({
    content: 'Historical role assignment for the org chart.',
    sourceId: 'org-chart-2024',
    structuredFacts: [
      {
        subject: 'Acme-Corp',
        predicate: 'ledBy',
        object: 'Jordan',
        validFrom: '2024-01-01T00:00:00Z',
        validTo: '2024-12-31T23:59:59Z',
      },
      // A second, CURRENT fact with metadata for filterMetadata below.
      {
        subject: 'Acme-Corp',
        predicate: 'ledBy',
        object: 'Riley',
        validFrom: '2025-01-01T00:00:00Z',
      },
    ],
    metadata: { docType: 'org-chart', status: 'active' },
    skipExtraction: true,
  });

  // ── k: cap result count ───────────────────────────────────────────────────
  console.log('[04] recall() with k=1');
  const kCapped = await mem.recall('who leads Acme-Corp?', { k: 1 });
  assert.ok(Array.isArray(kCapped));
  assert.ok(kCapped.length <= 1, `k=1 must cap results, got ${kCapped.length}`);

  // ── RetrievedContext.facts / RetrievedFact shape ─────────────────────────
  console.log('[04] recall() and inspect RetrievedContext.facts (ADR-074/TD-116)');
  const results = await mem.recall('who leads Acme-Corp?', { k: 10 });
  assert.ok(Array.isArray(results));
  console.log(`[04] recall returned ${results.length} entities`);
  let sawAnyFact = false;
  for (const r of results) {
    assert.ok(Array.isArray(r.facts), 'RetrievedContext.facts must be an array');
    for (const f of r.facts) {
      sawAnyFact = true;
      console.log(`  - fact: "${f.fact}" (subject=${f.subject} predicate=${f.predicate} object=${f.object})`);
      assert.equal(typeof f.fact, 'string');
      assert.equal(typeof f.subject, 'string');
      assert.equal(typeof f.predicate, 'string');
      assert.equal(typeof f.object, 'string');
      assert.equal(typeof f.objectIsEntity, 'boolean');
      assert.ok(!Number.isNaN(Date.parse(f.validAt)), 'validAt must be RFC-3339');
      assert.ok(!Number.isNaN(Date.parse(f.recordedAt)), 'recordedAt must be RFC-3339');
      assert.equal(typeof f.confidence, 'number');
      assert.ok(Array.isArray(f.sourceEpisodeIds));
      assert.equal(typeof f.score, 'number');
    }
  }
  assert.ok(sawAnyFact, 'at least one recalled entity must carry connected facts');

  // ── asOf: bi-temporal point-in-time filter ────────────────────────────────
  console.log('[04] recall() with asOf inside the historical window (mid-2024)');
  const midHistory = await mem.recall('who leads Acme-Corp?', {
    asOf: '2024-06-01T00:00:00Z',
    k: 10,
  });
  const historyFacts = midHistory.flatMap((r) => r.facts.map((f) => f.object));
  console.log('[04] asOf=2024-06-01 facts objects:', historyFacts);
  assert.ok(historyFacts.includes('Jordan'), 'Jordan must be visible as-of mid-2024');
  assert.ok(!historyFacts.includes('Riley'), 'Riley (valid from 2025) must NOT be visible as-of mid-2024');

  console.log('[04] recall() with asOf far in the future (2030) — before either fact was created is a different case; here confirm the window has closed for Jordan');
  const afterHistory = await mem.recall('who leads Acme-Corp?', {
    asOf: '2025-06-01T00:00:00Z',
    k: 10,
  });
  const afterFacts = afterHistory.flatMap((r) => r.facts.map((f) => f.object));
  console.log('[04] asOf=2025-06-01 facts objects:', afterFacts);
  assert.ok(afterFacts.includes('Riley'), 'Riley must be visible as-of mid-2025');
  assert.ok(!afterFacts.includes('Jordan'), 'Jordan (valid_to end of 2024) must NOT be visible as-of mid-2025');

  // ── filterMetadata: post-filter on episode metadata ──────────────────────
  console.log('[04] recall() with filterMetadata=[{docType: "org-chart"}]');
  const filtered = await mem.recall('who leads Acme-Corp?', {
    k: 10,
    filterMetadata: [{ key: 'docType', value: 'org-chart' }],
  });
  assert.ok(Array.isArray(filtered));
  console.log(`[04] filterMetadata narrowed to ${filtered.length} result(s)`);

  // Invalid (empty) key rejects at await time.
  await assert.rejects(
    () => mem.recall('q', { filterMetadata: [{ key: '', value: 'x' }] }),
    (err) => {
      assert.ok(err instanceof Error);
      console.log('[04] empty filterMetadata key rejected as documented:', err.message);
      return true;
    },
  );

  // ── rerankK: accepted regardless of build features — documented no-op ───
  //
  // This local build's default features are `["content-search"]` only (no
  // `rerank`) — per index.d.ts's own doc comment, `rerankK` is "a documented
  // no-op unless the binding was compiled with the rerank feature." Exercise
  // it here to confirm the call succeeds (accepted, not rejected) rather
  // than silently assuming the doc's claim.
  console.log('[04] recall() with rerankK=50 on a build WITHOUT the rerank feature (documented no-op)');
  const rerankAttempt = await mem.recall('who leads Acme-Corp?', { k: 5, rerankK: 50 });
  assert.ok(Array.isArray(rerankAttempt), 'rerankK must be accepted (no-op), not rejected, on a non-rerank build');

  console.log('[04] PASS');
} finally {
  await mem.close();
  try { fs.unlinkSync(dbPath); } catch { /* best-effort cleanup */ }
}
