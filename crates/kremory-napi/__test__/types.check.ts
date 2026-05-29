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

// Re-export so tsc keeps the symbol resolved (avoids dead-code stripping
// affecting the import graph).
export { _openOpts, _ingestOpts, _recallOpts, _exerciseSurface };
