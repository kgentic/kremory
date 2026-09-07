// kremory-napi episode parity tests (node:test runner).
//
// Phase G / TD-003: verifies that JsEpisode returned by recallBySourceId
// exposes source_id + content_hash (previously always null before G-2/G-3).
//
// F49 (public-docs-and-api-surface-audit phase1-findings.md): this file was
// written against the pre-ADR-034 API shape (`getBySourceId(slug, { namespace
// }) `, `remember({ ..., sourceKind: 'Document' })`) and never updated when
// `Memory.remember()` was unified — the method is `recallBySourceId(sourceId,
// namespace?: string)` (a plain string second arg, not an options object),
// and `RememberOptions` has no `sourceKind` field. Renamed to the current
// shape below; behaviour asserted is unchanged.
//
// REQUIRES: `pnpm build` (or `npm run build`) in `crates/kremory-napi/` first.
// The native .node binary must be present at index.js.
//
// Run: `node --test __test__/episode-parity.test.mjs`

import { test } from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

function tmpDbPath(tag) {
  return path.join(
    os.tmpdir(),
    `kremory-napi-episode-parity-${tag}-${Date.now()}-${process.pid}.db`,
  );
}

function cleanup(p) {
  try {
    if (fs.existsSync(p)) fs.unlinkSync(p);
  } catch {}
}

// ── 1. source_id parity ──────────────────────────────────────────────────────
//
// JsEpisode.sourceId must equal the slug used at ingest.
// Regression: before G-2/G-3, episode_to_js hardcoded sourceId from the query
// arg rather than the struct field; now it uses ep.source_id from the Episode struct.

test('recallBySourceId: JsEpisode.sourceId matches ingest slug', async () => {
  const dbPath = tmpDbPath('source-id-parity');
  let mem;
  try {
    mem = await Memory.open(dbPath, { defaultNamespace: 'parity-test' });

    const slug = 'episode-parity-test-slug-001';

    await mem.remember({
      content: 'Phase G napi parity test — source_id field.',
      namespace: 'parity-test',
      sourceId: slug,
    });

    const episodes = await mem.recallBySourceId(slug, 'parity-test');
    assert.ok(Array.isArray(episodes), 'recallBySourceId must return an array');
    assert.ok(episodes.length > 0, 'recallBySourceId must return at least one episode');

    const ep = episodes[0];
    assert.equal(
      ep.sourceId,
      slug,
      `JsEpisode.sourceId must equal ingest slug "${slug}" (regression: was always null before G-2/G-3)`,
    );

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 2. content_hash parity ───────────────────────────────────────────────────
//
// JsEpisode.contentHash must be a 64-hex-char SHA-256 string.
// Regression: before G-1 Migration 011, the column did not exist and the
// field was hardcoded None (null in JS).

test('recallBySourceId: JsEpisode.contentHash is 64-hex SHA-256', async () => {
  const dbPath = tmpDbPath('content-hash-parity');
  let mem;
  try {
    mem = await Memory.open(dbPath, { defaultNamespace: 'parity-test' });

    const slug = 'episode-parity-test-slug-002';

    await mem.remember({
      content: 'Phase G napi parity test — content_hash field.',
      namespace: 'parity-test',
      sourceId: slug,
    });

    const episodes = await mem.recallBySourceId(slug, 'parity-test');
    assert.ok(episodes.length > 0, 'must have at least one episode');

    const ep = episodes[0];
    assert.ok(
      ep.contentHash != null,
      `JsEpisode.contentHash must not be null after Migration 011 backfill (regression: was always null before G-1)`,
    );
    assert.equal(
      ep.contentHash.length,
      64,
      `contentHash must be 64 hex chars (SHA-256); got "${ep.contentHash}"`,
    );
    assert.match(
      ep.contentHash,
      /^[0-9a-f]{64}$/,
      `contentHash must be lowercase hex; got "${ep.contentHash}"`,
    );

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 3. JsEpisode shape completeness ─────────────────────────────────────────
//
// All expected fields must be present on the returned JsEpisode object.

test('recallBySourceId: JsEpisode has all expected fields', async () => {
  const dbPath = tmpDbPath('shape-check');
  let mem;
  try {
    mem = await Memory.open(dbPath, { defaultNamespace: 'parity-test' });

    const slug = 'episode-parity-shape-check-001';

    await mem.remember({
      content: 'Phase G napi parity test — shape check.',
      namespace: 'parity-test',
      sourceId: slug,
    });

    const episodes = await mem.recallBySourceId(slug, 'parity-test');
    assert.ok(episodes.length > 0, 'must have at least one episode');

    const ep = episodes[0];

    // Fields that must be PRESENT (this test set them at ingest, or the
    // substrate always populates them). `sourceUri` is deliberately excluded
    // from this list: it's an `Option<String>` on the substrate
    // (`JsEpisode.source_uri`, convert.rs) and — unlike a JS field explicitly
    // set to `null` — napi-rs's `#[napi(object)]` derive omits a `None`
    // `Option<T>` field from the object's own keys ENTIRELY when unset, so
    // `'sourceUri' in ep` is the wrong check for an optional field; checked
    // separately below via the nullish-tolerant idiom this codebase already
    // uses for the same field (`smoke.test.mjs`'s
    // `ep.sourceUri == null` assertion), not `in`.
    const requiredFields = ['id', 'sourceId', 'content', 'timestamp', 'contentHash'];
    for (const field of requiredFields) {
      assert.ok(
        field in ep,
        `JsEpisode must have field "${field}"; got keys: ${Object.keys(ep).join(', ')}`,
      );
    }
    assert.ok(
      ep.sourceUri == null || typeof ep.sourceUri === 'string',
      `JsEpisode.sourceUri must be nullish or a string when not set at ingest; got ${JSON.stringify(ep.sourceUri)}`,
    );

    assert.equal(typeof ep.id, 'number', 'id must be a number');
    assert.equal(typeof ep.content, 'string', 'content must be a string');
    assert.equal(typeof ep.timestamp, 'string', 'timestamp must be a string');
    assert.ok(!Number.isNaN(Date.parse(ep.timestamp)), `timestamp must be parseable as date: ${ep.timestamp}`);

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});
