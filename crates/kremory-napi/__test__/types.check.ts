// TS-shape smoke. This file is NEVER executed — only type-checked via
// `tsc --noEmit __test__/types.check.ts`. Catches drift between the
// Rust-side napi exports and the generated `index.d.ts`.
//
// If a kremory facade method changes its TS surface, this file will fail
// to compile — surfacing the drift before consumers hit it.
//
// Run: `pnpm test:types`

import { Memory } from '../index.js';
import type {
  OpenOptions,
  RememberOptions,
  RecallOptions,
  IngestResult,
  RetrievedContext,
  StructuredFact,
  Episode,
  DreamSummary,
  ConsolidationOpsRan,
  TypeProposal,
  DreamOptions,
  MetadataFilter,
  BatchOptions,
  IngestStatusResult,
  DreamStatusResult,
  BatchStatus,
  CancelOutcome,
} from '../index.js';

// ── Surface contracts ──────────────────────────────────────────────────────

// OpenOptions must accept embeddingDim + defaultNamespace.
const _openOpts: OpenOptions = {
  embeddingDim: 384,
  defaultNamespace: 'check',
};

// OpenOptions must accept withEmbedder callback (ADR-030 Tier-2 BYOM).
const _openOptsWithEmbedder: OpenOptions = {
  embeddingDim: 256,
  defaultNamespace: 'byom',
  withEmbedder: async (text: string): Promise<number[]> => {
    return Array.from({ length: 256 }, (_, i) => (i / 256) * (text.length / 100));
  },
};

// withEmbedder must also accept null and undefined (optional field).
const _openOptsNullEmbedder: OpenOptions = { withEmbedder: null };
const _openOptsUndefinedEmbedder: OpenOptions = { withEmbedder: undefined };

// RememberOptions must accept content + namespace + referenceTime.
const _opts: RememberOptions = {
  content: 'check content',
  namespace: 'check',
  referenceTime: '2026-05-29T00:00:00Z',
};

// RecallOptions must accept the ADR-029c surface: in_namespaces,
// best_effort, per_namespace_top_k. If napi-rs camel-casing changes these,
// this declaration fails to type-check.
const _recallOpts: RecallOptions = {
  k: 5,
  namespace: 'check',
  inNamespaces: ['a', 'b'],
  bestEffort: true,
  perNamespaceTopK: 20,
  asOf: '2026-05-29T00:00:00Z',
};

// IngestResult shape (returned by ingest).
function _checkIngestResult(r: IngestResult): void {
  const _id: string = r.episodeEntityId;
  const _ts: string = r.committedAt;
  void _id;
  void _ts;
}

// RetrievedContext shape (per-row of recall). namespace must be present
// on the type (ADR-029c). entityTypeId + entityTypeName added by TD-013 Phase 8.
function _checkRetrieved(r: RetrievedContext): void {
  const _entityId: string = r.entityId;
  const _entityName: string = r.entityName;
  const _summary: string = r.summary;
  const _score: number = r.score;
  const _refs: string[] = r.sourceRefs;
  const _incomplete: boolean = r.incomplete;
  // TD-013 Phase 8: entity type fields (additive, always present).
  const _typeId: number = r.entityTypeId;
  const _typeName: string = r.entityTypeName;
  // namespace is Option<String> on the Rust side -> `string | null | undefined`
  // in napi-rs generated TS. Accept all three.
  const _ns: string | null | undefined = r.namespace;
  void _entityId;
  void _entityName;
  void _summary;
  void _score;
  void _refs;
  void _incomplete;
  void _typeId;
  void _typeName;
  void _ns;
}

// Memory async-method contract: open/ingest/recall/close must return
// Promises of the documented shapes.
async function _exerciseSurface(): Promise<void> {
  const mem: Memory = await Memory.open('/tmp/x.db', _openOpts);
  const ingest: IngestResult = await mem.remember(_opts);
  _checkIngestResult(ingest);

  const results: RetrievedContext[] = await mem.recall('q', _recallOpts);
  for (const r of results) _checkRetrieved(r);

  await mem.close();
}

// ── B1: RememberOptions shape ────────────────────────────────────────────

const _episodeDraft: RememberOptions = {
  content: 'episode body',
  sourceId: 'doc-001',
  sourceUri: 'path/to/doc.md',
  metadata: { docType: 'adr', status: 'active' },
  namespace: 'check',
};

const _episodeDraftMinimal: RememberOptions = {
  content: 'bare episode, no source_id',
};

// ── B1: IngestResult now includes warnings ───────────────────────────────

function _checkIngestResultV2(r: IngestResult): void {
  const _id: string = r.episodeEntityId;
  const _ts: string = r.committedAt;
  const _warnings: string[] = r.warnings;
  void _id;
  void _ts;
  void _warnings;
}

// ── B5: Episode shape ────────────────────────────────────────────────────

function _checkEpisode(ep: Episode): void {
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

// ── B6: DreamSummary + DreamOptions shapes ───────────────────────────────

function _checkDreamSummary(s: DreamSummary): void {
  const _cu: number = s.communitiesUpdated;
  // ADR-073 Tier-0 D5: the old `crossEpisodeMerges` field was split into a
  // would-merge / did-merge pair. Assert both.
  const _cewm: number = s.crossEpisodeWouldMerge;
  const _cem: number = s.crossEpisodeMerged;
  const _sr: number = s.supersessionsRecorded;
  const _fa: number = s.factsArchived;
  // ADR-073 Tier-0 D1b: per-op ran-signal disambiguates "op disabled" from
  // "op ran, found nothing" on the all-zero consolidation counts above.
  const _ran: ConsolidationOpsRan = s.consolidationOpsRan;
  const _ranCommunity: boolean = _ran.community;
  const _ranCrossEpisode: boolean = _ran.crossEpisode;
  const _ranArchival: boolean = _ran.archival;
  const _ranSupersession: boolean = _ran.supersessionSweep;
  // ADR-071 §Item 4a / TD-060: budget-ceiling trip flag.
  const _be: boolean = s.budgetExhausted;
  const _dm: number = s.durationMs;
  // Reconciliation-pass accounting (mirrors substrate DreamSummary exactly).
  const _td: TypeProposal[] = s.typesDiscovered;
  const _er: number = s.entitiesReclassified;
  const _ar: number = s.aliasesResolved;
  const _cm: number = s.canonicalizationMerges;
  const _an: number = s.acronymNicknameMerges;
  const _trm: number = s.typeRegistryMerges;
  const _ccc: number = s.consistencyCheckCorrected;
  const _warnings: string[] = s.warnings;
  void _cu;
  void _cewm;
  void _cem;
  void _sr;
  void _fa;
  void _ranCommunity;
  void _ranCrossEpisode;
  void _ranArchival;
  void _ranSupersession;
  void _be;
  void _dm;
  void _td;
  void _er;
  void _ar;
  void _cm;
  void _an;
  void _trm;
  void _ccc;
  void _warnings;
}

const _dreamOpts: DreamOptions = { namespace: 'check' };

// ── B9: RecallOptions now includes filterMetadata ───────────────────────

const _recallOptsWithFilter: RecallOptions = {
  k: 5,
  namespace: 'check',
  filterMetadata: [
    { key: 'docType', value: 'adr' },
    { key: 'status', value: 'active' },
  ],
};

const _metadataFilter: MetadataFilter = {
  key: 'docType',
  value: 'adr',
};

// ── B2/B3/B5/B6/B7/B8: Memory method surface ───────────────────────────

async function _exerciseNewSurface(): Promise<void> {
  const mem: Memory = await Memory.open('/tmp/x.db', _openOpts);

  // B1: remember (replaces ingestEpisode)
  const draft: RememberOptions = _episodeDraft;
  const r1: IngestResult = await mem.remember(draft);
  _checkIngestResultV2(r1);

  // B2: updateEpisodeMetadata
  const updated: number = await mem.updateEpisodeMetadata('doc-001', { status: 'active' });
  void updated;

  // B3: updateSourceUri
  const uriRows: number = await mem.updateSourceUri('doc-001', 'path/new.md');
  void uriRows;

  // B5: recallBySourceId
  const episodes: Episode[] = await mem.recallBySourceId('doc-001', 'check');
  for (const ep of episodes) _checkEpisode(ep);

  // B6: dream
  const summary: DreamSummary = await mem.dream(_dreamOpts);
  _checkDreamSummary(summary);

  // B7: forget
  const deleted: number = await mem.forget('doc-001', 'check');
  void deleted;

  // B9: recall with filterMetadata
  const filteredResults = await mem.recall('query', _recallOptsWithFilter);
  void filteredResults;

  await mem.close();
}

// ── v0.1.8 surface: new types + 9 new methods ───────────────────────────

// A1. BatchOptions literal — episodes array + optional batchId.
const _batchOpts: BatchOptions = {
  episodes: [
    { content: 'episode one', namespace: 'v018' },
    { content: 'episode two', namespace: 'v018', skipExtraction: true },
  ],
  batchId: 'batch-v018-001',
};

// A1b. BatchOptions minimal (batchId optional).
const _batchOptsMinimal: BatchOptions = {
  episodes: [{ content: 'bare episode' }],
};

// A2. IngestStatusResult exercises all status variant strings + optional errorMessage.
const _ingestStatusPending: IngestStatusResult = { status: 'pending' };
const _ingestStatusExtracting: IngestStatusResult = { status: 'extracting' };
const _ingestStatusDeduplicating: IngestStatusResult = { status: 'deduplicating' };
const _ingestStatusInvalidating: IngestStatusResult = { status: 'invalidating' };
const _ingestStatusComplete: IngestStatusResult = { status: 'complete' };
const _ingestStatusFailed: IngestStatusResult = { status: 'failed', errorMessage: 'extraction timed out' };

// A3. DreamStatusResult — same shape as IngestStatusResult.
const _dreamStatusPending: DreamStatusResult = { status: 'pending' };
const _dreamStatusProcessing: DreamStatusResult = { status: 'processing' };
const _dreamStatusComplete: DreamStatusResult = { status: 'complete' };
const _dreamStatusFailed: DreamStatusResult = { status: 'failed', errorMessage: 'dream failed' };

// A4. BatchStatus literal — all numeric fields present.
const _batchStatus: BatchStatus = {
  total: 10,
  completed: 8,
  skipped: 1,
  failed: 1,
};

// A5. CancelOutcome literal — cancelledPhase, rolledBack, partial.
const _cancelOutcomeEnrichment: CancelOutcome = {
  cancelledPhase: 'enrichment',
  rolledBack: true,
  partial: [],
};
const _cancelOutcomeConsolidation: CancelOutcome = {
  cancelledPhase: 'consolidation',
  rolledBack: false,
  partial: ['entity-uuid-001', 'entity-uuid-002'],
};

// A7. IngestResult.runId is string | null | undefined (Quinn M-01 fix).
function _checkIngestResultRunId(r: IngestResult): void {
  const _runId: string | null | undefined = r.runId;
  void _runId;
}

// A6. Type-check the 9 new method signatures on Memory.
async function _exerciseV018NewSurface(): Promise<void> {
  const mem: Memory = await Memory.open('/tmp/v018.db', _openOpts);

  const _rememberBatch: (opts: BatchOptions) => Promise<IngestResult[]> = mem.rememberBatch.bind(mem);
  const _statusOf: (commitId: string) => Promise<IngestStatusResult> = mem.statusOf.bind(mem);
  const _awaitEnrichment: (commitId: string, timeoutMs: number) => Promise<IngestStatusResult> = mem.awaitEnrichment.bind(mem);
  const _awaitDream: (handleId: string, timeoutMs: number) => Promise<DreamStatusResult> = mem.awaitDream.bind(mem);
  const _awaitBatch: (batchId: string, timeoutMs: number) => Promise<BatchStatus> = mem.awaitBatch.bind(mem);
  const _cancel: (commitId: string) => Promise<CancelOutcome> = mem.cancel.bind(mem);
  const _cancelDream: (handleId: string) => Promise<CancelOutcome> = mem.cancelDream.bind(mem);
  const _registerNamespace: (namespace: string) => Promise<void> = mem.registerNamespace.bind(mem);
  const _upgradeNamespacePolicy: (namespace: string) => Promise<void> = mem.upgradeNamespacePolicy.bind(mem);

  void _rememberBatch;
  void _statusOf;
  void _awaitEnrichment;
  void _awaitDream;
  void _awaitBatch;
  void _cancel;
  void _cancelDream;
  void _registerNamespace;
  void _upgradeNamespacePolicy;

  await mem.close();
}

// Re-export so tsc keeps the symbol resolved (avoids dead-code stripping
// affecting the import graph).
export {
  _openOpts,
  _openOptsWithEmbedder,
  _openOptsNullEmbedder,
  _openOptsUndefinedEmbedder,
  _opts,
  _recallOpts,
  _exerciseSurface,
  _episodeDraft,
  _episodeDraftMinimal,
  _dreamOpts,
  _recallOptsWithFilter,
  _metadataFilter,
  _exerciseNewSurface,
  // v0.1.8
  _batchOpts,
  _batchOptsMinimal,
  _ingestStatusPending,
  _ingestStatusExtracting,
  _ingestStatusDeduplicating,
  _ingestStatusInvalidating,
  _ingestStatusComplete,
  _ingestStatusFailed,
  _dreamStatusPending,
  _dreamStatusProcessing,
  _dreamStatusComplete,
  _dreamStatusFailed,
  _batchStatus,
  _cancelOutcomeEnrichment,
  _cancelOutcomeConsolidation,
  _exerciseV018NewSurface,
};
