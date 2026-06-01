// TS-shape smoke. This file is NEVER executed — only type-checked via
// `tsc --noEmit __test__/types.check.ts`. Catches drift between the
// Rust-side napi exports and the generated `index.d.ts`.
//
// If a kremory facade method changes its TS surface, this file will fail
// to compile — surfacing the drift before consumers hit it.
//
// Run: `pnpm test:types`

import { JsMemory } from '../index.js';
import type {
  JsOpenOptions,
  JsIngestOptions,
  JsRecallOptions,
  JsIngestResult,
  JsRetrievedContext,
  JsEpisodeDraft,
  JsEpisode,
  JsDreamSummary,
  JsDreamOpts,
  JsMetadataFilter,
} from '../index.js';

// ── Surface contracts ──────────────────────────────────────────────────────

// JsOpenOptions must accept embeddingDim + defaultNamespace.
const _openOpts: JsOpenOptions = {
  embeddingDim: 384,
  defaultNamespace: 'check',
};

// JsIngestOptions must accept namespace + referenceTime + contentType.
const _ingestOpts: JsIngestOptions = {
  namespace: 'check',
  referenceTime: '2026-05-29T00:00:00Z',
  contentType: 'chat',
};

// JsRecallOptions must accept the ADR-029c surface: in_namespaces,
// best_effort, per_namespace_top_k. If napi-rs camel-casing changes these,
// this declaration fails to type-check.
const _recallOpts: JsRecallOptions = {
  k: 5,
  namespace: 'check',
  inNamespaces: ['a', 'b'],
  bestEffort: true,
  perNamespaceTopK: 20,
  asOf: '2026-05-29T00:00:00Z',
};

// JsIngestResult shape (returned by ingest).
function _checkIngestResult(r: JsIngestResult): void {
  const _id: string = r.episodeEntityId;
  const _ts: string = r.committedAt;
  void _id;
  void _ts;
}

// JsRetrievedContext shape (per-row of recall). namespace must be present
// on the type (ADR-029c).
function _checkRetrieved(r: JsRetrievedContext): void {
  const _entityId: string = r.entityId;
  const _entityName: string = r.entityName;
  const _summary: string = r.summary;
  const _score: number = r.score;
  const _refs: string[] = r.sourceRefs;
  const _incomplete: boolean = r.incomplete;
  // namespace is Option<String> on the Rust side -> `string | null | undefined`
  // in napi-rs generated TS. Accept all three.
  const _ns: string | null | undefined = r.namespace;
  void _entityId;
  void _entityName;
  void _summary;
  void _score;
  void _refs;
  void _incomplete;
  void _ns;
}

// JsMemory async-method contract: open/ingest/recall/close must return
// Promises of the documented shapes.
async function _exerciseSurface(): Promise<void> {
  const mem: JsMemory = await JsMemory.open('/tmp/x.db', _openOpts);
  const ingest: JsIngestResult = await mem.ingest('hi', _ingestOpts);
  _checkIngestResult(ingest);

  const results: JsRetrievedContext[] = await mem.recall('q', _recallOpts);
  for (const r of results) _checkRetrieved(r);

  await mem.close();
}

// ── B1: JsEpisodeDraft shape ───────────────────────────────────────────────

const _episodeDraft: JsEpisodeDraft = {
  content: 'episode body',
  sourceId: 'doc-001',
  sourceUri: 'path/to/doc.md',
  metadata: { docType: 'adr', status: 'active' },
  namespace: 'check',
};

const _episodeDraftMinimal: JsEpisodeDraft = {
  content: 'bare episode, no source_id',
};

// ── B1: JsIngestResult now includes warnings ───────────────────────────────

function _checkIngestResultV2(r: JsIngestResult): void {
  const _id: string = r.episodeEntityId;
  const _ts: string = r.committedAt;
  const _warnings: string[] = r.warnings;
  void _id;
  void _ts;
  void _warnings;
}

// ── B5: JsEpisode shape ────────────────────────────────────────────────────

function _checkEpisode(ep: JsEpisode): void {
  const _id: number = ep.id;
  const _sid: string | null | undefined = ep.sourceId;
  const _uri: string | null | undefined = ep.sourceUri;
  const _content: string = ep.content;
  const _ts: string = ep.timestamp;
  const _st: string | null | undefined = ep.sourceType;
  const _meta: Record<string, unknown> | null | undefined = ep.metadata;
  const _hash: string | null | undefined = ep.contentHash;
  void _id;
  void _sid;
  void _uri;
  void _content;
  void _ts;
  void _st;
  void _meta;
  void _hash;
}

// ── B6: JsDreamSummary + JsDreamOpts shapes ───────────────────────────────

function _checkDreamSummary(s: JsDreamSummary): void {
  const _cu: number = s.communitiesUpdated;
  const _cem: number = s.crossEpisodeMerges;
  const _sr: number = s.supersessionsRecorded;
  const _fa: number = s.factsArchived;
  const _dm: number = s.durationMs;
  void _cu;
  void _cem;
  void _sr;
  void _fa;
  void _dm;
}

const _dreamOpts: JsDreamOpts = { namespace: 'check' };

// ── B9: JsRecallOptions now includes filterMetadata ───────────────────────

const _recallOptsWithFilter: JsRecallOptions = {
  k: 5,
  namespace: 'check',
  filterMetadata: [
    { key: 'docType', value: 'adr' },
    { key: 'status', value: 'active' },
  ],
};

const _metadataFilter: JsMetadataFilter = {
  key: 'docType',
  value: 'adr',
};

// ── B2/B3/B5/B6/B7/B8: JsMemory method surface ───────────────────────────

async function _exerciseNewSurface(): Promise<void> {
  const mem: JsMemory = await JsMemory.open('/tmp/x.db', _openOpts);

  // B1: ingestEpisode
  const draft: JsEpisodeDraft = _episodeDraft;
  const r1: JsIngestResult = await mem.ingestEpisode(draft);
  _checkIngestResultV2(r1);

  // B2: updateMetadata
  const updated: number = await mem.updateMetadata('doc-001', { status: 'active' });
  void updated;

  // B3: updateUri
  const uriRows: number = await mem.updateUri('doc-001', 'path/new.md');
  void uriRows;

  // B5: getBySourceId
  const episodes: JsEpisode[] = await mem.getBySourceId('doc-001', 'check');
  for (const ep of episodes) _checkEpisode(ep);

  // B6: dream
  const summary: JsDreamSummary = await mem.dream(_dreamOpts);
  _checkDreamSummary(summary);

  // B7: forget
  const deleted: number = await mem.forget('doc-001', 'check');
  void deleted;

  // B8: reindex (stub — always rejects)
  try {
    await mem.reindex();
  } catch (_e) {
    // expected
  }

  // B9: recall with filterMetadata
  const filteredResults = await mem.recall('query', _recallOptsWithFilter);
  void filteredResults;

  await mem.close();
}

// Re-export so tsc keeps the symbol resolved (avoids dead-code stripping
// affecting the import graph).
export {
  _openOpts,
  _ingestOpts,
  _recallOpts,
  _exerciseSurface,
  _episodeDraft,
  _episodeDraftMinimal,
  _dreamOpts,
  _recallOptsWithFilter,
  _metadataFilter,
  _exerciseNewSurface,
};
