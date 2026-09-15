// kremory-napi reversibility surface — supersede, unsupersede,
// mutationHistory, listMutations, the unified undo() dispatcher,
// editEntity/undoEntityEdit, deleteEntity/undoDeleteEntity,
// deleteFact/undoDeleteFact, restoreArchivedFact.
//
// ── Finding F40 (see docs/specs/public-docs-and-api-surface-audit/phase1-findings.md) ──
//
// `supersede` / `deleteFact` / `restoreArchivedFact` / `unsupersede` all
// require a numeric `factId` as their PRIMARY argument. There is NO public
// path — on the Node binding OR the Rust facade — to discover a fact's
// numeric id: `RetrievedFact` (returned by `recall()`) carries no `id` field
// (verified against `crates/kremory/src/memory/types.rs`'s `RetrievedFact`
// struct, which genuinely has no such field — not a napi-binding-only gap),
// and there is no `listFacts`-shaped method anywhere. The ONLY way a real
// consumer legitimately learns a factId today is by already having received
// one from a PRIOR mutation call that echoes it back (e.g. `deleteFact`'s
// own return value, or `restoreArchivedFact`'s `restoredFactId`) — which is
// circular for the FIRST call on a fact nobody has touched yet.
//
// This example demonstrates the mutation methods by relying on a
// deterministic-but-unsupported implementation detail: SQLite
// auto-increment row ids are sequential starting at 1 in a FRESH database,
// so inserting exactly one `structuredFacts` entry per `remember()` call, in
// a brand-new db file, gives fact ids 1, 2, 3, ... in call order. **This is
// NOT a documented or supported pattern — do not rely on it in real code.**
// It is used here ONLY to exercise these methods' shapes deterministically
// in an example, and is called out explicitly rather than silently assumed.
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED (v0.4.1) — this imports the
// LOCALLY BUILT native module, not a published npm package.
//
// REQUIRES: native module built (`pnpm build:debug`);
//   OLLAMA_HOST=http://localhost:11434 (or OPENAI_API_KEY / ANTHROPIC_API_KEY).
//
// Run: `node examples/06-reversibility-and-undo.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

const NS = 'reversibility-demo';
const dbPath = path.join(os.tmpdir(), `kremory-example-reversibility-${Date.now()}.db`);

console.log('[06] opening Memory (fresh db — fact ids below are deterministic BECAUSE this is a fresh db, see header comment)');
const mem = await Memory.open(dbPath, { defaultNamespace: NS });

try {
  // fact id 1 — used by supersede/unsupersede.
  await mem.remember({
    content: 'Supersede demo seed.',
    structuredFacts: [{ subject: 'SupersedeDemo', predicate: 'status', object: 'Draft' }],
    skipExtraction: true,
  });
  // fact id 2 — used by direct deleteFact + restoreArchivedFact.
  await mem.remember({
    content: 'Delete-fact demo seed 1.',
    structuredFacts: [{ subject: 'DeleteFactDemo1', predicate: 'status', object: 'Live' }],
    skipExtraction: true,
  });
  // fact id 3 — used by deleteFact + undoDeleteFact.
  await mem.remember({
    content: 'Delete-fact demo seed 2.',
    structuredFacts: [{ subject: 'DeleteFactDemo2', predicate: 'status', object: 'Live' }],
    skipExtraction: true,
  });
  // Entities for editEntity / deleteEntity flows (subjects become entities).
  await mem.remember({
    content: 'Edit-entity demo seed.',
    structuredFacts: [{ subject: 'EditDemo', predicate: 'status', object: 'Live' }],
    skipExtraction: true,
  });
  await mem.remember({
    content: 'Undo-dispatch demo seed.',
    structuredFacts: [{ subject: 'UndoDispatchDemo', predicate: 'status', object: 'Live' }],
    skipExtraction: true,
  });
  await mem.remember({
    content: 'Delete-entity demo seed.',
    structuredFacts: [{ subject: 'DeleteEntityDemo', predicate: 'status', object: 'Live' }],
    skipExtraction: true,
  });

  // ── supersede: bound a fact's world-time valid_to window ─────────────────
  const supersedeFactId = 1;
  console.log(`[06] supersede(${supersedeFactId}, future date)`);
  const supersedeResult = await mem.supersede(
    supersedeFactId,
    '2099-01-01T00:00:00Z',
    'demo: bounding for a future date, not yet closed',
  );
  console.log('[06] supersede result:', supersedeResult);
  assert.equal(supersedeResult.outcome, 'bounded');
  assert.equal(supersedeResult.retired, 0, 'a future-dated bound must not retire anything inline');

  // supersede on an unknown factId is an honest not_found, not a crash.
  const notFoundSupersede = await mem.supersede(999_999, '2030-01-01T00:00:00Z');
  console.log('[06] supersede on unknown factId:', notFoundSupersede);
  assert.equal(notFoundSupersede.outcome, 'not_found');

  // ── unsupersede: clear the bound just set ─────────────────────────────────
  console.log(`[06] unsupersede(${supersedeFactId})`);
  const unsupersedeResult = await mem.unsupersede(supersedeFactId);
  console.log('[06] unsupersede result:', unsupersedeResult);
  assert.equal(unsupersedeResult.outcome, 'cleared');
  assert.equal(unsupersedeResult.factId, supersedeFactId);
  assert.equal(unsupersedeResult.clearedValidTo, true);

  // Calling again on an already-cleared fact is an honest no-op.
  const alreadyClearedResult = await mem.unsupersede(supersedeFactId);
  console.log('[06] unsupersede again (already cleared):', alreadyClearedResult);
  assert.equal(alreadyClearedResult.outcome, 'not_superseded');

  // ── deleteFact + restoreArchivedFact (direct, not via undoDeleteFact) ────
  const directFactId = 2;
  console.log(`[06] deleteFact(${directFactId})`);
  const deleteOutcome1 = await mem.deleteFact(directFactId);
  console.log('[06] deleteFact result:', deleteOutcome1);
  assert.equal(deleteOutcome1.factId, directFactId);
  assert.equal(deleteOutcome1.alreadyUndone, false);

  console.log(`[06] restoreArchivedFact(${directFactId}) — direct restore, not via undoDeleteFact`);
  const restoreOutcome = await mem.restoreArchivedFact(directFactId);
  console.log('[06] restoreArchivedFact result:', restoreOutcome);
  assert.equal(restoreOutcome.restoredFactId, directFactId);
  assert.equal(restoreOutcome.alreadyLive, false);

  // ── Finding F44 (logged in phase1-findings.md) — FIXED 2026-09-07 ────────
  //
  // Originally: `restoreArchivedFact`'s documented idempotency
  // (`already_live: true` on a repeat call) was unreachable via the natural
  // double-call sequence — the first check was `facts_archive` presence,
  // but a successful restore's own last step deletes that row, so the
  // second call threw "no facts_archive row" before the idempotent branch
  // ever ran. RE-VERIFIED 2026-09-15 (re-running this example against the
  // current build): this no longer reproduces. Fixed by reordering
  // `restore_archived_txn` (`core/dream/provenance/reversal.rs`) to check
  // `facts` (already-live) FIRST; a genuinely bogus id still falls through
  // to the `facts_archive` check and errors loudly. F44 is stale; retained
  // here only as history.
  console.log(`[06] restoreArchivedFact(${directFactId}) AGAIN — must be idempotent`);
  const restoreAgain = await mem.restoreArchivedFact(directFactId);
  console.log('[06] restoreArchivedFact (again) result:', restoreAgain);
  assert.equal(restoreAgain.restoredFactId, directFactId);
  assert.equal(restoreAgain.alreadyLive, true, 'second call must report already_live per the documented contract');

  // ── deleteFact + undoDeleteFact (via the per-kind undo method) ───────────
  const undoFactId = 3;
  console.log(`[06] deleteFact(${undoFactId})`);
  const deleteOutcome2 = await mem.deleteFact(undoFactId);
  console.log('[06] deleteFact result:', deleteOutcome2);
  const factMutationId = deleteOutcome2.mutationId;
  assert.equal(typeof factMutationId, 'number');

  console.log(`[06] undoDeleteFact(${factMutationId})`);
  const undoFactOutcome = await mem.undoDeleteFact(factMutationId);
  console.log('[06] undoDeleteFact result:', undoFactOutcome);
  assert.equal(undoFactOutcome.factRestored, true);
  assert.equal(undoFactOutcome.factId, undoFactId);

  // Idempotent: undoing again is an honest no-op.
  const undoFactAgain = await mem.undoDeleteFact(factMutationId);
  console.log('[06] undoDeleteFact again (idempotent):', undoFactAgain);
  assert.equal(undoFactAgain.alreadyUndone, true);

  // ── editEntity + undoEntityEdit (per-kind undo) ──────────────────────────
  console.log('[06] editEntity("EditDemo", { newId: "EditDemoRenamed" })');
  const editOutcome = await mem.editEntity('EditDemo', { newId: 'EditDemoRenamed', namespace: NS });
  console.log('[06] editEntity result:', editOutcome);
  assert.equal(editOutcome.entityId, 'EditDemoRenamed');
  assert.equal(editOutcome.rekeyed, true);
  assert.equal(editOutcome.retyped, false);
  const editMutationId = editOutcome.mutationId;

  console.log(`[06] undoEntityEdit(${editMutationId})`);
  const undoEditOutcome = await mem.undoEntityEdit(editMutationId);
  console.log('[06] undoEntityEdit result:', undoEditOutcome);
  assert.equal(undoEditOutcome.entityId, 'EditDemo', 'undo must restore the original entity id');

  // ── the unified undo() dispatcher (ADR-073 DX R1/R2) ─────────────────────
  console.log('[06] editEntity("UndoDispatchDemo", { typeId: 0 }) then undo() via the unified dispatcher');
  const dispatchEdit = await mem.editEntity('UndoDispatchDemo', { typeId: 0, namespace: NS });
  console.log('[06] editEntity (retype) result:', dispatchEdit);
  assert.equal(dispatchEdit.retyped, true);
  assert.equal(dispatchEdit.rekeyed, false);

  const unifiedUndo = await mem.undo(dispatchEdit.mutationId, NS);
  console.log('[06] undo() unified dispatcher result:', unifiedUndo);
  assert.equal(unifiedUndo.kind, 'edit_entity');
  assert.ok(unifiedUndo.editEntity, 'undo() must populate the .editEntity field when kind === "edit_entity"');
  assert.equal(unifiedUndo.unmerge, undefined, 'undo() must leave non-matching kind fields absent');
  assert.equal(unifiedUndo.deleteEntity, undefined);
  assert.equal(unifiedUndo.deleteFact, undefined);

  // ── deleteEntity + undoDeleteEntity ───────────────────────────────────────
  console.log('[06] deleteEntity("DeleteEntityDemo")');
  const deleteEntityOutcome = await mem.deleteEntity('DeleteEntityDemo', NS);
  console.log('[06] deleteEntity result:', deleteEntityOutcome);
  assert.equal(deleteEntityOutcome.entityId, 'DeleteEntityDemo');

  console.log(`[06] undoDeleteEntity(${deleteEntityOutcome.mutationId})`);
  const undoDeleteEntityOutcome = await mem.undoDeleteEntity(deleteEntityOutcome.mutationId);
  console.log('[06] undoDeleteEntity result:', undoDeleteEntityOutcome);
  assert.equal(undoDeleteEntityOutcome.entityId, 'DeleteEntityDemo');

  // ── mutationHistory: inspect what touched one entity ─────────────────────
  console.log('[06] mutationHistory("DeleteEntityDemo", NS)');
  const history = await mem.mutationHistory('DeleteEntityDemo', NS);
  assert.ok(Array.isArray(history));
  console.log(`[06] mutationHistory returned ${history.length} record(s)`);
  for (const rec of history) {
    console.log(`  - mutationId=${rec.mutationId} kind=${rec.kind} undone=${rec.undone} summary="${rec.summary}"`);
    assert.equal(typeof rec.mutationId, 'number');
    assert.equal(typeof rec.kind, 'string');
    assert.ok(!Number.isNaN(Date.parse(rec.createdAt)));
    assert.equal(typeof rec.undone, 'boolean');
    assert.equal(rec.groupId, NS);
    assert.ok(Array.isArray(rec.affectedEntities));
    assert.equal(typeof rec.summary, 'string');
  }
  assert.ok(history.length >= 1, 'DeleteEntityDemo must have at least one logged mutation');

  // ── listMutations: global inspect surface + filters ──────────────────────
  console.log('[06] listMutations({ namespace: NS })');
  const allMutations = await mem.listMutations({ namespace: NS, includeUndone: true });
  assert.ok(Array.isArray(allMutations));
  console.log(`[06] listMutations (includeUndone) returned ${allMutations.length} record(s)`);
  assert.ok(allMutations.length >= 4, 'expect at least the edit/undo/delete/undo-delete mutations logged above');

  console.log('[06] listMutations({ namespace: NS, kind: "entity_edit" })');
  const editKindOnly = await mem.listMutations({ namespace: NS, kind: 'entity_edit', includeUndone: true });
  assert.ok(Array.isArray(editKindOnly));
  for (const rec of editKindOnly) assert.equal(rec.kind, 'entity_edit');
  console.log(`[06] entity_edit-only listMutations returned ${editKindOnly.length} record(s)`);

  // Unknown kind tag rejects.
  await assert.rejects(
    () => mem.listMutations({ kind: 'not-a-real-kind' }),
    (err) => {
      assert.ok(err instanceof Error);
      console.log('[06] unknown mutation kind rejected as documented:', err.message);
      return true;
    },
  );

  // ── unmerge — NOT exercised at runtime here, see examples/10-unmerge.ts ──
  //
  // `unmerge` reverses a prior entity-merge (a `dream()` cross-episode
  // reconciliation decision) by its mutationId. Triggering a REAL merge
  // deterministically requires dream()'s LLM/similarity-driven entity
  // resolution to decide two entities are aliases of the same real-world
  // thing and fuse them in "apply" mode — not something this example set can
  // reliably reproduce as a fast, deterministic demo. See
  // `examples/10-unmerge.ts` for the type-checked (not runtime-executed)
  // shape verification, and this file's own README entry for the rationale.
  console.log('[06] unmerge: skipped at runtime (non-deterministic precondition) — see examples/10-unmerge.ts');

  console.log('[06] PASS');
} finally {
  await mem.close();
  try { fs.unlinkSync(dbPath); } catch { /* best-effort cleanup */ }
}
