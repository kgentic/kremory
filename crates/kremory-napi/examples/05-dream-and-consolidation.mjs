// kremory-napi dream phase + consolidation — dream() with the full
// DreamOptions surface (community/archival/supersession/crossEpisodeMode/
// budgets), runDreamPassSync(), ghostEpisodes(), assertEntityType().
//
// Uses `skipExtraction: true` for setup so the demonstration is fast and
// deterministic; dream() itself is real (no mocking) — it runs the actual
// zero-LLM consolidation ops (community detection, archival, supersession
// sweep) against the real graph.
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED (v0.4.1) — this imports the
// LOCALLY BUILT native module, not a published npm package.
//
// REQUIRES: native module built (`pnpm build:debug`);
//   OLLAMA_HOST=http://localhost:11434 (or OPENAI_API_KEY / ANTHROPIC_API_KEY).
//
// Run: `node examples/05-dream-and-consolidation.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

const dbPath = path.join(os.tmpdir(), `kremory-example-dream-${Date.now()}.db`);

console.log('[05] opening Memory');
const mem = await Memory.open(dbPath, { defaultNamespace: 'dream-demo' });

try {
  // ── Seed: one episode with pinned facts (creates real entities) ─────────
  console.log('[05] remember() with structuredFacts to seed a deterministic entity');
  await mem.remember({
    content: 'Seed episode for dream-phase demonstration.',
    structuredFacts: [{ subject: 'DreamDemoEntity', predicate: 'hasRole', object: 'Example' }],
    skipExtraction: true,
  });

  // ── A "ghost" episode: skipExtraction with NO structuredFacts -> no facts ─
  console.log('[05] remember() with skipExtraction and no facts — a ghost episode candidate');
  await mem.remember({
    content: 'Ghost episode: ingested but no facts ever extracted for it.',
    skipExtraction: true,
  });

  // ── ghostEpisodes: episode ids where Phase 1 succeeded, Phase 2 produced no facts ─
  console.log('[05] ghostEpisodes("dream-demo")');
  const ghosts = await mem.ghostEpisodes('dream-demo');
  assert.ok(Array.isArray(ghosts), 'ghostEpisodes must return an array');
  console.log(`[05] ghost episode ids: ${JSON.stringify(ghosts)}`);
  for (const id of ghosts) assert.equal(typeof id, 'number');

  // ── assertEntityType: pin an entity as ConsumerPinned ────────────────────
  console.log('[05] assertEntityType("DreamDemoEntity", 0, "dream-demo") — pin to the "Entity" catch-all type id');
  await mem.assertEntityType('DreamDemoEntity', 0, 'dream-demo');

  // ── runDreamPassSync: lower-level single-pass entry point ────────────────
  console.log('[05] runDreamPassSync() — Phase C DoD C7 lower-level dream entry point');
  const passSummary = await mem.runDreamPassSync({
    includeTypeDiscovery: false,
    confidenceThreshold: 0.5,
    reclassifyHighConfThreshold: 0.7,
  });
  console.log('[05] runDreamPassSync summary:', passSummary);
  assert.equal(typeof passSummary.durationMs, 'number');
  assert.ok(Array.isArray(passSummary.warnings));

  // ── dream(): full DreamOptions surface ────────────────────────────────────
  console.log('[05] dream() with the full consolidation-control surface');
  const summary = await mem.dream({
    namespace: 'dream-demo',
    includeCommunityDetection: true,
    includeFactArchival: true,
    includeSupersessionSweep: true,
    includeSupersessionLlmNominate: false,
    crossEpisodeMode: 'shadow', // computes would-merge decisions, fuses nothing
    archiveGraceDays: 90,
    netMutationWarnFloor: 500,
    consolidationBudgetTokens: 50_000,
  });
  console.log('[05] dream summary:', summary);

  // D1b: per-op ran-signal disambiguates "op disabled" from "op ran, found nothing".
  assert.equal(typeof summary.consolidationOpsRan.community, 'boolean');
  assert.equal(typeof summary.consolidationOpsRan.crossEpisode, 'boolean');
  assert.equal(typeof summary.consolidationOpsRan.archival, 'boolean');
  assert.equal(typeof summary.consolidationOpsRan.supersessionSweep, 'boolean');
  assert.ok(
    summary.consolidationOpsRan.community,
    'includeCommunityDetection: true must set consolidationOpsRan.community',
  );
  assert.ok(
    summary.consolidationOpsRan.archival,
    'includeFactArchival: true must set consolidationOpsRan.archival',
  );
  assert.ok(
    summary.consolidationOpsRan.supersessionSweep,
    'includeSupersessionSweep: true must set consolidationOpsRan.supersessionSweep',
  );

  // D5: would-merge vs did-merge split. crossEpisodeMode="shadow" -> merged
  // must be 0 regardless of how many would-merge decisions were computed.
  assert.equal(
    summary.crossEpisodeMerged,
    0,
    'crossEpisodeMode="shadow" must never actually fuse entities (crossEpisodeMerged == 0)',
  );
  assert.ok(summary.crossEpisodeWouldMerge >= 0);

  assert.equal(typeof summary.budgetExhausted, 'boolean');
  assert.equal(typeof summary.durationMs, 'number');
  assert.ok(Array.isArray(summary.typesDiscovered));
  assert.equal(typeof summary.entitiesReclassified, 'number');
  assert.equal(typeof summary.aliasesResolved, 'number');
  assert.equal(typeof summary.canonicalizationMerges, 'number');
  assert.equal(typeof summary.acronymNicknameMerges, 'number');
  assert.equal(typeof summary.typeRegistryMerges, 'number');
  assert.equal(typeof summary.consistencyCheckCorrected, 'number');
  assert.ok(Array.isArray(summary.warnings));

  // ── dream() with an unknown crossEpisodeMode string rejects (loud parse) ──
  await assert.rejects(
    () => mem.dream({ namespace: 'dream-demo', crossEpisodeMode: 'not-a-real-mode' }),
    (err) => {
      assert.ok(err instanceof Error);
      console.log('[05] unknown crossEpisodeMode rejected as documented:', err.message);
      return true;
    },
  );

  console.log('[05] PASS');
} finally {
  await mem.close();
  try { fs.unlinkSync(dbPath); } catch { /* best-effort cleanup */ }
}
