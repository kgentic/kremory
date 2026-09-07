// kremory-napi `unmerge` — reverse a prior entity-merge by its mutationId.
//
// This file is NEVER executed — only type-checked, mirroring
// `__test__/types.check.ts`'s own established convention (see its header
// comment: "This file is NEVER executed — only type-checked via `tsc
// --noEmit __test__/types.check.ts`. Catches drift between the Rust-side
// napi exports and the generated `index.d.ts`.").
//
// Why `unmerge` specifically gets this treatment instead of a real runtime
// example (unlike everything else in this directory): triggering a REAL
// entity merge requires `dream()`'s cross-episode reconciliation pass to
// make an LLM/similarity-driven decision that two entities are aliases of
// the same real-world thing, then fuse them in `crossEpisodeMode: "apply"`.
// That decision is not something this example set can reliably reproduce as
// a fast, deterministic demo — it depends on real model output for entity
// similarity judgement, which is exactly the kind of thing
// `smoke-one-before-batch-llm-validation` discipline says not to assume
// works from a single hand-constructed scenario. Verifying the SHAPE of the
// call is still valuable, so it is checked here instead.
//
// Run: `npx tsc --noEmit --target es2022 --module nodenext --moduleResolution nodenext --strict --skipLibCheck examples/10-unmerge.ts`
// (mirrors `package.json`'s own `test:types` script, pointed at this file
// instead of `__test__/types.check.ts`.)

import { Memory } from '../index.js';
import type { UnmergeOutcome, MutationRecord, MutationFilter } from '../index.js';

function checkUnmergeOutcome(o: UnmergeOutcome): void {
  const _restoredEntity: string = o.restoredEntity;
  const _keeper: string = o.keeper;
  const _factsRepointed: number = o.factsRepointed;
  const _edgesRestored: number = o.edgesRestored;
  const _entitiesReopened: number = o.entitiesReopened;
  const _nogoodRecorded: boolean = o.nogoodRecorded;
  const _alreadyUndone: boolean = o.alreadyUndone;
  void _restoredEntity;
  void _keeper;
  void _factsRepointed;
  void _edgesRestored;
  void _entitiesReopened;
  void _nogoodRecorded;
  void _alreadyUndone;
}

function checkMutationRecord(r: MutationRecord): void {
  const _mutationId: number = r.mutationId;
  const _kind: string = r.kind;
  const _createdAt: string = r.createdAt;
  const _undone: boolean = r.undone;
  const _groupId: string = r.groupId;
  const _affectedEntities: string[] = r.affectedEntities;
  const _summary: string = r.summary;
  void _mutationId;
  void _kind;
  void _createdAt;
  void _undone;
  void _groupId;
  void _affectedEntities;
  void _summary;
}

const _mutationFilterForMerges: MutationFilter = {
  kind: 'entity_merge',
  includeUndone: true,
};

async function _exerciseUnmergeSurface(): Promise<void> {
  const mem: Memory = await Memory.open('/tmp/unmerge-typecheck.db');

  // The realistic call shape a consumer follows: find an entity_merge
  // mutation via listMutations/mutationHistory, then reverse it by id.
  const merges: MutationRecord[] = await mem.listMutations(_mutationFilterForMerges);
  for (const rec of merges) checkMutationRecord(rec);

  if (merges.length > 0) {
    const outcome: UnmergeOutcome = await mem.unmerge(merges[0].mutationId);
    checkUnmergeOutcome(outcome);

    // Also reachable via the unified dispatcher — `undo()` routes an
    // `entity_merge`-kind mutationId to the same unmerge path internally.
    const dispatched = await mem.undo(merges[0].mutationId);
    if (dispatched.kind === 'unmerge' && dispatched.unmerge) {
      checkUnmergeOutcome(dispatched.unmerge);
    }
  }

  await mem.close();
}

export { checkUnmergeOutcome, checkMutationRecord, _mutationFilterForMerges, _exerciseUnmergeSurface };
