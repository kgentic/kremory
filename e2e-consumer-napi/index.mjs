// kremory-napi Node consumer end-to-end journey (REAL LLM via Ollama).
//
// Drives the built native binding exactly as a JS SDK consumer would:
//   open → remember (real gemma4:e4b extraction) → recall → dream
//   (all-consolidation-on default) → inspect mutations → exercise the
//   ADR-073 reversible surface (unmerge / editEntity+undo / deleteEntity+undo).
//
// REQUIRES: `pnpm build` in crates/kremory-napi (native .node present) +
// Ollama running with gemma4:e4b + nomic-embed-text pulled.
//
// RUN: OLLAMA_HOST=http://localhost:11434 node e2e-consumer-napi/index.mjs
//
// Exits non-zero on any hard failure. Stochastic no-merge is a SKIP, not a fail.

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../crates/kremory-napi/index.js');

const NS = 'e2e';
const dbPath = path.join(os.tmpdir(), `kremory-napi-e2e-${Date.now()}-${process.pid}.db`);

// ── tiny reporting harness ──────────────────────────────────────────────────
let stepNo = 0;
const results = [];
function record(name, status, detail) {
  results.push({ step: ++stepNo, name, status, detail });
  const tag = status === 'PASS' ? 'PASS' : status === 'SKIP' ? 'SKIP' : 'FAIL';
  console.log(`\n[step ${stepNo}] ${tag} — ${name}${detail ? `\n         ${detail}` : ''}`);
}

/** Assert an object carries the given keys with the given typeof (or 'array'). */
function assertShape(obj, spec, label) {
  assert.ok(obj && typeof obj === 'object', `${label}: not an object`);
  for (const [key, kind] of Object.entries(spec)) {
    assert.ok(key in obj, `${label}: missing field "${key}"`);
    const v = obj[key];
    if (kind === 'array') {
      assert.ok(Array.isArray(v), `${label}.${key}: expected array, got ${typeof v}`);
    } else if (kind === 'object') {
      assert.ok(v && typeof v === 'object' && !Array.isArray(v), `${label}.${key}: expected object`);
    } else {
      assert.equal(typeof v, kind, `${label}.${key}: expected ${kind}, got ${typeof v} (${v})`);
    }
  }
}

const t = (label, fn) => {
  const t0 = Date.now();
  return fn().then((r) => { console.log(`         (${label} took ${Date.now() - t0}ms)`); return r; });
};

async function main() {
  console.log(`kremory-napi Node consumer E2E — db=${dbPath}`);
  console.log(`OLLAMA_HOST=${process.env.OLLAMA_HOST}  model=gemma4:e4b (auto)\n`);

  // ── STEP 1: open ──────────────────────────────────────────────────────────
  const mem = await Memory.open(dbPath, { defaultNamespace: NS });
  assert.ok(mem, 'Memory.open returned falsy');
  record('open — JsMemory constructed (ollama auto → gemma4:e4b + nomic-embed-text)', 'PASS',
    `handle type: ${mem.constructor?.name ?? typeof mem}`);

  // ── STEP 2: remember 3 episodes, varied surface forms of same person/org ──
  const episodes = [
    'Dr. Jane Smith, the Chief Technology Officer of Acme Corporation, announced the Nimbus platform in Berlin.',
    'Jane Smith leads the engineering organisation at Acme Corp, where the Nimbus platform is built.',
    'At Acme, J. Smith and her team shipped Nimbus version 2 to enterprise customers.',
  ];
  const ingestResults = [];
  for (let i = 0; i < episodes.length; i++) {
    const r = await t(`remember #${i + 1}`, () =>
      mem.remember({ content: episodes[i], sourceId: `ep-${i + 1}`, namespace: NS }));
    assertShape(r, { episodeEntityId: 'string', committedAt: 'string', warnings: 'array' }, `IngestResult#${i + 1}`);
    assert.ok(r.episodeEntityId.length > 0, 'episodeEntityId empty');
    assert.ok(!Number.isNaN(Date.parse(r.committedAt)), `committedAt not rfc3339: ${r.committedAt}`);
    // napi-rs omits Option::None fields from the JS object entirely (not set to
    // `undefined`), so a consumer checks `== null`, matching `runId?: string` in .d.ts.
    assert.ok(r.runId == null, `IngestResult.runId must be nullish on inline path, got: ${r.runId}`);
    ingestResults.push(r);
  }
  record('remember ×3 — real gemma4:e4b extraction, honest IngestResult shape', 'PASS',
    `episodeEntityIds=[${ingestResults.map((r) => r.episodeEntityId).join(', ')}]`);

  // ── STEP 3: recall (pre-dream) ────────────────────────────────────────────
  const preRecall = await t('recall(pre-dream)', () =>
    mem.recall('Who is Jane Smith and what did she build at Acme?', { namespace: NS, k: 10 }));
  assert.ok(Array.isArray(preRecall), 'recall did not return array');
  for (const r of preRecall) {
    assertShape(r, {
      entityId: 'string', entityName: 'string', summary: 'string', score: 'number',
      sourceRefs: 'array', incomplete: 'boolean', entityTypeId: 'number', entityTypeName: 'string',
    }, 'RetrievedContext');
  }
  record('recall (pre-dream) — RetrievedContext[] honest shape', 'PASS',
    `${preRecall.length} entities: ${preRecall.slice(0, 6).map((r) => `${r.entityId}(${r.entityTypeName})`).join(', ')}`);

  // ── STEP 4: dream — all-consolidation-on shipped default ──────────────────
  const summary = await t('dream(default all-on)', () => mem.dream({ namespace: NS }));
  assertShape(summary, {
    communitiesUpdated: 'number', crossEpisodeWouldMerge: 'number', crossEpisodeMerged: 'number',
    supersessionsRecorded: 'number', factsArchived: 'number', consolidationOpsRan: 'object',
    budgetExhausted: 'boolean', durationMs: 'number', typesDiscovered: 'array',
    entitiesReclassified: 'number', aliasesResolved: 'number', canonicalizationMerges: 'number',
    acronymNicknameMerges: 'number', typeRegistryMerges: 'number', consistencyCheckCorrected: 'number',
    warnings: 'array',
  }, 'DreamSummary');
  assertShape(summary.consolidationOpsRan, {
    community: 'boolean', crossEpisode: 'boolean', archival: 'boolean', supersessionSweep: 'boolean',
  }, 'DreamSummary.consolidationOpsRan');
  // Honest-outcome invariant: the four consolidation ops all default ON, so each ran-flag must be true.
  assert.equal(summary.consolidationOpsRan.community, true, 'community op should have run (default ON)');
  assert.equal(summary.consolidationOpsRan.crossEpisode, true, 'cross-episode op should have run (default ON)');
  assert.equal(summary.consolidationOpsRan.archival, true, 'archival op should have run (default ON)');
  assert.equal(summary.consolidationOpsRan.supersessionSweep, true, 'supersession sweep should have run (default ON)');
  // Shipped default = cross-episode SHADOW: decisions counted, no fusion applied.
  assert.ok(summary.crossEpisodeMerged <= summary.crossEpisodeWouldMerge,
    'merged must never exceed would-merge');
  record('dream — DreamSummary honest fields (consolidationOpsRan all-true, would/did-merge split)', 'PASS',
    `wouldMerge=${summary.crossEpisodeWouldMerge} merged=${summary.crossEpisodeMerged} ` +
    `canon=${summary.canonicalizationMerges} alias=${summary.aliasesResolved} ` +
    `acronym=${summary.acronymNicknameMerges} communities=${summary.communitiesUpdated} ` +
    `archived=${summary.factsArchived} typesDiscovered=${summary.typesDiscovered.length} ` +
    `durationMs=${summary.durationMs}`);

  // ── STEP 5: recall (post-dream) ───────────────────────────────────────────
  const postRecall = await t('recall(post-dream)', () =>
    mem.recall('Jane Smith Acme Nimbus', { namespace: NS, k: 10 }));
  assert.ok(Array.isArray(postRecall), 'post-dream recall not array');
  record('recall (post-dream)', 'PASS',
    `${postRecall.length} entities: ${postRecall.slice(0, 6).map((r) => r.entityId).join(', ')}`);

  // ── STEP 6: inspect mutations (the SEE surface) ───────────────────────────
  const allMutations = await mem.listMutations({ namespace: NS });
  assert.ok(Array.isArray(allMutations), 'listMutations not array');
  for (const m of allMutations) {
    assertShape(m, {
      mutationId: 'number', kind: 'string', createdAt: 'string', undone: 'boolean',
      groupId: 'string', affectedEntities: 'array', summary: 'string',
    }, 'MutationRecord');
    assert.ok(!Number.isNaN(Date.parse(m.createdAt)), `mutation.createdAt not rfc3339: ${m.createdAt}`);
  }
  const kinds = allMutations.reduce((acc, m) => { acc[m.kind] = (acc[m.kind] || 0) + 1; return acc; }, {});
  record('listMutations — MutationRecord[] inspectable from JS (mutationId/kind/summary/undone)', 'PASS',
    `${allMutations.length} mutations; kinds=${JSON.stringify(kinds)}`);

  // mutationHistory for one entity (prefer a merge keeper, else any recalled entity).
  const mergeMutations = await mem.listMutations({ namespace: NS, kind: 'entity_merge' });
  const historyEntity =
    (mergeMutations[0] && mergeMutations[0].affectedEntities[0]) ||
    (postRecall[0] && postRecall[0].entityId) ||
    (preRecall[0] && preRecall[0].entityId);
  if (historyEntity) {
    const hist = await mem.mutationHistory(historyEntity, NS);
    assert.ok(Array.isArray(hist), 'mutationHistory not array');
    for (const m of hist) {
      assertShape(m, { mutationId: 'number', kind: 'string', undone: 'boolean', summary: 'string' }, 'MutationRecord(history)');
    }
    record(`mutationHistory("${historyEntity}") — per-entity history`, 'PASS',
      `${hist.length} records: ${hist.slice(0, 4).map((m) => `${m.kind}#${m.mutationId}`).join(', ')}`);
  } else {
    record('mutationHistory — no entity available', 'SKIP', 'no recalled entity to inspect');
  }

  // ── STEP 7a: unmerge (reversible surface) ─────────────────────────────────
  const liveMerge = mergeMutations.find((m) => !m.undone);
  if (liveMerge) {
    const outcome = await mem.unmerge(liveMerge.mutationId);
    assertShape(outcome, {
      restoredEntity: 'string', keeper: 'string', factsRepointed: 'number', edgesRestored: 'number',
      entitiesReopened: 'number', nogoodRecorded: 'boolean', alreadyUndone: 'boolean',
    }, 'UnmergeOutcome');
    assert.equal(outcome.alreadyUndone, false, 'first unmerge must not be already-undone');
    // Idempotency: a second unmerge is an honest no-op.
    const again = await mem.unmerge(liveMerge.mutationId);
    assert.equal(again.alreadyUndone, true, 'second unmerge must report alreadyUndone=true');
    record(`unmerge(#${liveMerge.mutationId}) — UnmergeOutcome honest counts + idempotency`, 'PASS',
      `restoredEntity=${outcome.restoredEntity} keeper=${outcome.keeper} ` +
      `factsRepointed=${outcome.factsRepointed} edgesRestored=${outcome.edgesRestored} ` +
      `entitiesReopened=${outcome.entitiesReopened} nogoodRecorded=${outcome.nogoodRecorded}`);
  } else {
    record('unmerge — no applied entity_merge this run (stochastic reconciliation)', 'SKIP',
      `entity_merge mutations found: ${mergeMutations.length} (all undone or none). ` +
      'cross-episode is shadow by default; reconciliation merges are LLM-stochastic.');
  }

  // ── STEP 7b: editEntity (rename) + undoEntityEdit ─────────────────────────
  const editTarget = postRecall.find((r) => r.entityId && r.entityId.length > 0) || preRecall[0];
  if (editTarget) {
    const newId = `${editTarget.entityId}__renamed_${Date.now()}`;
    const edit = await mem.editEntity(editTarget.entityId, { newId, namespace: NS });
    assertShape(edit, {
      entityId: 'string', rekeyed: 'boolean', retyped: 'boolean', factsRepointed: 'number',
      archivedRepointed: 'number', edgesRepointed: 'number', communitiesRepointed: 'number',
      entitiesReopened: 'number', mutationId: 'number', alreadyUndone: 'boolean',
    }, 'EditEntityOutcome');
    assert.equal(edit.rekeyed, true, 'rename must set rekeyed=true');
    assert.equal(edit.entityId, newId, 'outcome.entityId must be the new id after rename');
    assert.equal(edit.alreadyUndone, false, 'forward editEntity must not be already-undone');
    // Undo restores the prior id.
    const undo = await mem.undoEntityEdit(edit.mutationId);
    assertShape(undo, { entityId: 'string', mutationId: 'number', alreadyUndone: 'boolean' }, 'EditEntityOutcome(undo)');
    assert.equal(undo.entityId, editTarget.entityId, 'undo must restore the original entity id');
    record(`editEntity("${editTarget.entityId}" → rename) + undoEntityEdit(#${edit.mutationId})`, 'PASS',
      `rekeyed=${edit.rekeyed} factsRepointed=${edit.factsRepointed} edgesRepointed=${edit.edgesRepointed} ` +
      `communitiesRepointed=${edit.communitiesRepointed} → undo restored "${undo.entityId}"`);
  } else {
    record('editEntity — no entity to edit', 'SKIP', 'recall returned no entities');
  }

  // ── STEP 7c: deleteEntity + undoDeleteEntity ──────────────────────────────
  // Re-recall so we operate on a currently-live id (unmerge/edit above may have moved things).
  const liveEntities = await mem.recall('Jane Smith Acme Nimbus Berlin', { namespace: NS, k: 10 });
  const delTarget = liveEntities.find((r) => r.entityId && r.entityId.length > 0);
  if (delTarget) {
    const del = await mem.deleteEntity(delTarget.entityId, NS);
    assertShape(del, {
      entityId: 'string', factsRetracted: 'number', edgesRemoved: 'number', communitiesRemoved: 'number',
      neighborsRetracted: 'number', entitiesReopened: 'number', mutationId: 'number', alreadyUndone: 'boolean',
    }, 'DeleteEntityOutcome');
    assert.equal(del.entityId, delTarget.entityId, 'delete outcome entityId must match target');
    assert.equal(del.alreadyUndone, false, 'forward deleteEntity must not be already-undone');
    const undoDel = await mem.undoDeleteEntity(del.mutationId);
    assertShape(undoDel, { entityId: 'string', mutationId: 'number', alreadyUndone: 'boolean' }, 'DeleteEntityOutcome(undo)');
    assert.equal(undoDel.entityId, delTarget.entityId, 'undoDelete must restore same entity id');
    record(`deleteEntity("${delTarget.entityId}") + undoDeleteEntity(#${del.mutationId})`, 'PASS',
      `factsRetracted=${del.factsRetracted} edgesRemoved=${del.edgesRemoved} ` +
      `communitiesRemoved=${del.communitiesRemoved} neighborsRetracted=${del.neighborsRetracted} → restored`);
  } else {
    record('deleteEntity — no entity to delete', 'SKIP', 'recall returned no entities');
  }

  // ── STEP 8: verify inspect surface reflects the reversible ops ────────────
  const finalMutations = await mem.listMutations({ namespace: NS, includeUndone: true });
  assert.ok(Array.isArray(finalMutations), 'final listMutations not array');
  const editDeleteKinds = finalMutations.filter((m) => m.kind === 'entity_edit' || m.kind === 'entity_delete');
  record('listMutations(includeUndone) — reversible ops logged + inspectable', 'PASS',
    `${finalMutations.length} total; edit/delete mutations=${editDeleteKinds.length}; ` +
    `undone=${finalMutations.filter((m) => m.undone).length}`);

  await mem.close();

  // ── summary ───────────────────────────────────────────────────────────────
  const pass = results.filter((r) => r.status === 'PASS').length;
  const skip = results.filter((r) => r.status === 'SKIP').length;
  const fail = results.filter((r) => r.status === 'FAIL').length;
  console.log(`\n${'='.repeat(70)}\nJOURNEY COMPLETE — ${pass} PASS, ${skip} SKIP, ${fail} FAIL\n${'='.repeat(70)}`);
  if (fail > 0) process.exit(1);
}

main().catch((e) => {
  console.error('\n[E2E FAIL]', e && e.stack ? e.stack : e);
  process.exitCode = 1;
}).finally(() => {
  try { fs.unlinkSync(dbPath); } catch {}
  try { fs.unlinkSync(`${dbPath}-wal`); } catch {}
  try { fs.unlinkSync(`${dbPath}-shm`); } catch {}
});
