// kremory-napi ingest options + batch — full `RememberOptions` surface
// (sourceId, sourceUri, metadata, referenceTime, structuredFacts,
// skipExtraction), updateEpisodeMetadata, updateSourceUri, recallBySourceId,
// rememberBatch + awaitBatch.
//
// Also demonstrates (and documents the limit of) `statusOf` / `awaitEnrichment`
// / `cancel` — see the block near the bottom of this file.
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED (v0.4.1) — this imports the
// LOCALLY BUILT native module, not a published npm package.
//
// REQUIRES: native module built (`pnpm build:debug`);
//   OLLAMA_HOST=http://localhost:11434 (or OPENAI_API_KEY / ANTHROPIC_API_KEY).
//
// Run: `node examples/03-ingest-batch-and-async-status.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

const dbPath = path.join(os.tmpdir(), `kremory-example-ingest-batch-${Date.now()}.db`);

console.log('[03] opening Memory');
const mem = await Memory.open(dbPath, { defaultNamespace: 'ingest-demo' });

try {
  // ── remember(): full RememberOptions surface ─────────────────────────────
  console.log('[03] remember() with sourceId/sourceUri/metadata/referenceTime/structuredFacts');
  const ingest = await mem.remember({
    content: 'Design notes: kremory-napi is the Node binding for kremory.',
    sourceId: 'design-note-001',
    sourceUri: 'docs/design/napi-binding.md',
    metadata: { docType: 'design', status: 'draft' },
    referenceTime: '2026-03-15T10:00:00Z',
    structuredFacts: [
      { subject: 'kremory-napi', predicate: 'bindsFor', object: 'kremory' },
    ],
    skipExtraction: true,
  });
  console.log('[03] ingest result:', ingest);
  assert.equal(typeof ingest.episodeEntityId, 'string');
  assert.ok(Array.isArray(ingest.warnings));

  // ── updateEpisodeMetadata: shallow merge by source_id ────────────────────
  console.log('[03] updateEpisodeMetadata("design-note-001", { status: "active", version: 2 })');
  const metaCount = await mem.updateEpisodeMetadata('design-note-001', {
    status: 'active',
    version: 2,
  });
  assert.ok(metaCount >= 1, `updateEpisodeMetadata must update >=1 row, got ${metaCount}`);
  console.log(`[03] updateEpisodeMetadata updated ${metaCount} row(s)`);

  // ── updateSourceUri ───────────────────────────────────────────────────────
  console.log('[03] updateSourceUri("design-note-001", "docs/design/napi-binding-v2.md")');
  const uriCount = await mem.updateSourceUri(
    'design-note-001',
    'docs/design/napi-binding-v2.md',
  );
  assert.ok(uriCount >= 1, `updateSourceUri must update >=1 row, got ${uriCount}`);

  // ── recallBySourceId: verify the writes landed ───────────────────────────
  console.log('[03] recallBySourceId("design-note-001")');
  const episodes = await mem.recallBySourceId('design-note-001');
  assert.ok(episodes.length >= 1, 'must find at least one episode');
  const ep = episodes[0];
  console.log('[03] episode:', ep);
  assert.equal(ep.sourceId, 'design-note-001');
  // `index.d.ts`'s `recallBySourceId` doc comment (its "Known gap" note)
  // claims `sourceUri` is ALWAYS null because "the substrate
  // recall_by_source_id query does not select that column." That is STALE —
  // `crates/kremory/src/facade/update.rs`'s `recall_by_source_id` SELECT has
  // included `source_uri` since TD-003 Phase G (its own inline comment says
  // so), and this run's actual returned value confirms it round-trips.
  // Logged as Finding F42 (docs/specs/.../phase1-findings.md) — a genuine
  // RULE-005 "Wrong"-class doc/code mismatch, discovered by actually running
  // this example rather than reading the .d.ts prose.
  assert.equal(
    ep.sourceUri,
    'docs/design/napi-binding-v2.md',
    'sourceUri DOES round-trip through recallBySourceId — the .d.ts "Known gap" note is stale (Finding F42)',
  );
  assert.equal(typeof ep.contentHash, 'string');
  assert.match(ep.contentHash, /^[0-9a-f]{64}$/, 'contentHash must be a 64-hex SHA-256 string');

  // ── rememberBatch + awaitBatch ────────────────────────────────────────────
  console.log('[03] rememberBatch() — 3 episodes, one namespace override');
  const batchId = 'demo-batch-001';
  const batchResults = await mem.rememberBatch({
    episodes: [
      { content: 'Batch episode one about apples.', skipExtraction: true },
      { content: 'Batch episode two about bridges.', skipExtraction: true },
      {
        content: 'Batch episode three, different namespace.',
        namespace: 'ingest-demo-overflow',
        skipExtraction: true,
      },
    ],
    batchId,
  });
  assert.equal(batchResults.length, 3, 'rememberBatch must return one result per input episode');
  for (const r of batchResults) {
    assert.equal(typeof r.episodeEntityId, 'string');
  }
  console.log(`[03] rememberBatch committed ${batchResults.length} episode(s)`);

  // ── awaitBatch: single- and multi-episode batches both complete ──
  //
  // Finding F43 (logged in phase1-findings.md) originally reported that
  // `awaitBatch` could never report done for 2+ episodes sharing a
  // `batchId` — `total` was only set by the first episode's `or_insert`,
  // never bumped by later episodes' `.and_modify`, so `is_done()` could
  // never match. RE-VERIFIED 2026-09-15 (re-running this example against
  // the current build): this no longer reproduces. It was fixed by TD-251
  // (`crates/kremory/src/memory/engine_handle.rs`,
  // `batch_status_increment_completed`/`_skipped` now bump `total` on both
  // the `.and_modify` and `.or_insert` arms, and batch registration itself
  // increments `total` per episode — see `batch_status_accumulates` and
  // `cancel_never_wedges_its_batch` on the Rust side). F43 is stale; the
  // finding is retained here only as history, not as a currently-reproducing
  // gap.
  console.log('[03] awaitBatch on a SINGLE-episode batch (this shape works)');
  const soloBatchId = 'demo-batch-solo';
  await mem.rememberBatch({
    episodes: [{ content: 'Solo batch episode.', skipExtraction: true }],
    batchId: soloBatchId,
  });
  const soloStatus = await mem.awaitBatch(soloBatchId, 10_000);
  console.log('[03] solo batch status (done):', soloStatus);
  assert.equal(soloStatus.total, 1);
  assert.equal(soloStatus.completed + soloStatus.skipped + soloStatus.failed, soloStatus.total);

  console.log(`[03] awaitBatch("${batchId}", 5000) on the 3-episode batch`);
  const multiStatus = await mem.awaitBatch(batchId, 5_000);
  console.log('[03] multi-episode batch status (done):', multiStatus);
  assert.equal(multiStatus.total, 3, 'total must count all 3 episodes sharing this batchId (TD-251)');
  assert.equal(
    multiStatus.completed + multiStatus.skipped + multiStatus.failed,
    multiStatus.total,
    'batch must report done — see TD-251',
  );

  // ── statusOf / awaitEnrichment / cancel: DOCUMENTED-UNREACHABLE in practice ─
  //
  // Both take a `commitId` that is only populated on `IngestResult.runId` when
  // the substrate spawns a BACKGROUND Phase-2 enrichment task (a `.no_wait()`
  // builder call on the Rust side). `RememberOptions` on this binding has no
  // field to request that path — `RememberRequest::no_wait` is listed in
  // `parity-skip.toml` as Rust-only, un-mirrored — and every `remember()` /
  // `rememberBatch()` call above returned `runId: null|undefined` (the
  // default INLINE path), confirmed by the existing
  // `__test__/smoke.test.mjs` "V018: IngestResult.runId is nullish..." test
  // and its own trailing TODO ("add statusOf/awaitEnrichment/cancel runtime
  // smoke once no_wait is exposed on JsRememberOptions ... parking lot").
  //
  // This is NOT a new finding — it is the same known, already-tracked gap —
  // but it means these three methods cannot be exercised with a REAL
  // background run from pure JS today. We still call `statusOf` with a
  // syntactically-valid-but-unknown UUID to exercise the method's shape.
  //
  // Actual (correct, documented-in-source) behaviour: an unrecognised
  // `commitId` does NOT reject — `graph_ingest_status` treats "no registry
  // entry" as "completed and evicted" (an intentional design choice, per its
  // own comment citing Hatchet/Temporal prior art: absence is non-failure
  // terminal success, not an error). So this resolves to `{ status:
  // "complete" }`, not a rejection.
  console.log('[03] statusOf() with an unknown commitId — demonstrates the (intentional) not-found-is-complete shape');
  const unknownStatus = await mem.statusOf('00000000-0000-0000-0000-000000000000');
  console.log('[03] statusOf on unknown commitId:', unknownStatus);
  assert.equal(
    unknownStatus.status,
    'complete',
    'an unrecognised commitId resolves to "complete" by design (absence = terminal success)',
  );

  console.log('[03] PASS');
} finally {
  await mem.close();
  try { fs.unlinkSync(dbPath); } catch { /* best-effort cleanup */ }
}
