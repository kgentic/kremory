// Compile-only check: a JS/TS consumer imports the generated TS surface and
// uses the ADR-073 reversible outcome types exactly as the runtime journey does.
// Run: tsc --noEmit --strict against this file. No runtime.

import type {
  IngestResult,
  DreamSummary,
  ConsolidationOpsRan,
  RetrievedContext,
  MutationRecord,
  UnmergeOutcome,
  EditEntityOutcome,
  DeleteEntityOutcome,
  OpenOptions,
  RememberOptions,
  DreamOptions,
  EditEntityOptions,
  MutationFilter,
} from '../crates/kremory-napi/index.js';
import { Memory } from '../crates/kremory-napi/index.js';

async function consume(): Promise<void> {
  const openOpts: OpenOptions = { defaultNamespace: 'e2e' };
  const mem = await Memory.open('/tmp/x.db', openOpts);

  const rememberOpts: RememberOptions = { content: 'x', sourceId: 's', namespace: 'e2e' };
  const ingest: IngestResult = await mem.remember(rememberOpts);
  const _eid: string = ingest.episodeEntityId;
  const _run: string | undefined | null = ingest.runId; // optional per .d.ts
  const _w: string[] = ingest.warnings;

  const recallResults: Array<RetrievedContext> = await mem.recall('q', { namespace: 'e2e', k: 10 });
  const _entityId: string = recallResults[0]?.entityId ?? '';

  const dreamOpts: DreamOptions = { namespace: 'e2e' };
  const summary: DreamSummary = await mem.dream(dreamOpts);
  const ops: ConsolidationOpsRan = summary.consolidationOpsRan;
  const _flags: boolean[] = [ops.community, ops.crossEpisode, ops.archival, ops.supersessionSweep];
  const _wm: number = summary.crossEpisodeWouldMerge;
  const _md: number = summary.crossEpisodeMerged;

  const filter: MutationFilter = { namespace: 'e2e', kind: 'entity_merge', includeUndone: true };
  const muts: Array<MutationRecord> = await mem.listMutations(filter);
  const _mid: number = muts[0]?.mutationId ?? 0;
  const _kind: string = muts[0]?.kind ?? '';
  const _summ: string = muts[0]?.summary ?? '';
  const _undone: boolean = muts[0]?.undone ?? false;

  const hist: Array<MutationRecord> = await mem.mutationHistory('e', 'e2e');
  void hist;

  const unmerge: UnmergeOutcome = await mem.unmerge(1);
  const _u: [string, string, number, boolean, boolean] =
    [unmerge.restoredEntity, unmerge.keeper, unmerge.edgesRestored, unmerge.nogoodRecorded, unmerge.alreadyUndone];

  const editOpts: EditEntityOptions = { newId: 'e_new', namespace: 'e2e' };
  const edit: EditEntityOutcome = await mem.editEntity('e', editOpts);
  const _e: [boolean, number, number] = [edit.rekeyed, edit.mutationId, edit.factsRepointed];
  await mem.undoEntityEdit(edit.mutationId);

  const del: DeleteEntityOutcome = await mem.deleteEntity('e', 'e2e');
  const _d: [number, number, number] = [del.factsRetracted, del.neighborsRetracted, del.mutationId];
  await mem.undoDeleteEntity(del.mutationId);

  await mem.close();
  void [_eid, _run, _w, _entityId, _flags, _wm, _md, _mid, _kind, _summ, _undone, _u, _e, _d];
}

void consume;
