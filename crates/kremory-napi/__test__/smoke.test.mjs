// kremory-napi smoke suite (node:test runner).
//
// REQUIRES: `pnpm install && pnpm build` (or `npm install && npm run build`)
// in `crates/kremory-napi/` first. The native .node binary must be present.
//
// Prerequisite: OLLAMA_HOST or OPENAI_API_KEY or ANTHROPIC_API_KEY in env
// for the env-auto provider detection path.
//
// Run: `node --test __test__/*.test.mjs`

import { test } from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { JsMemory } = require('../index.js');

function tmpDbPath(tag) {
  return path.join(os.tmpdir(), `kremory-smoke-${tag}-${Date.now()}-${process.pid}.db`);
}

function cleanup(p) {
  try {
    if (fs.existsSync(p)) fs.unlinkSync(p);
  } catch {}
}

// ── 1. Happy path ───────────────────────────────────────────────────────────

test('happy path: open → ingest → recall → close', async () => {
  const dbPath = tmpDbPath('happy');
  let mem;
  try {
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'smoke' });
    assert.ok(mem, 'JsMemory.open returned falsy');

    const ingest = await mem.ingest(
      'James prefers concise, data-driven technical decisions.',
      { namespace: 'smoke', contentType: 'chat' },
    );
    assert.equal(typeof ingest.episodeEntityId, 'string', 'episodeEntityId missing');
    assert.ok(ingest.episodeEntityId.length > 0, 'episodeEntityId empty');
    assert.ok(
      !Number.isNaN(Date.parse(ingest.committedAt)),
      `committedAt not rfc3339 parseable: ${ingest.committedAt}`,
    );

    const results = await mem.recall('what does James prefer?', { namespace: 'smoke', k: 5 });
    assert.ok(Array.isArray(results), 'recall did not return array');

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 2. Undefined per-call options (defaultNamespace fallback) ──────────────

test('undefined opts: ingest(text)/recall(query) use defaultNamespace from open', async () => {
  const dbPath = tmpDbPath('undef-opts');
  let mem;
  try {
    // Setting defaultNamespace at open is the contract for omitting per-call
    // namespace. Without either, kremory rejects with "namespace required".
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'undef' });

    // No opts — should not panic and should fall back to defaultNamespace.
    const ingest = await mem.ingest('Bare ingest, no options.');
    assert.equal(typeof ingest.episodeEntityId, 'string');

    const results = await mem.recall('bare recall');
    assert.ok(Array.isArray(results));

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 2b. Negative: no namespace anywhere → rejection ────────────────────────

test('no namespace anywhere: ingest rejects with namespace-required error', async () => {
  const dbPath = tmpDbPath('no-ns');
  let mem;
  try {
    mem = await JsMemory.open(dbPath); // no defaultNamespace

    await assert.rejects(
      async () => {
        await mem.ingest('orphan text'); // no per-call namespace either
      },
      (err) => {
        assert.ok(err instanceof Error);
        // kremory's error message contains "namespace required".
        assert.match(err.message, /namespace required/i);
        return true;
      },
    );

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 3. Open default no opts ─────────────────────────────────────────────────

test('JsMemory.open with no opts arg', async () => {
  const dbPath = tmpDbPath('open-no-opts');
  let mem;
  try {
    mem = await JsMemory.open(dbPath);
    assert.ok(mem);
    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 4. Error mapping: invalid path ──────────────────────────────────────────

test('error mapping: invalid path rejects with JS Error', async () => {
  // /dev/null/x.db is guaranteed-invalid on macOS + linux (cannot create
  // file under a non-directory). Should map to a JS Error, NOT panic the
  // process.
  await assert.rejects(
    async () => {
      await JsMemory.open('/dev/null/kremory-invalid.db');
    },
    (err) => {
      assert.ok(err instanceof Error, 'error not an instance of Error');
      assert.ok(
        typeof err.message === 'string' && err.message.length > 0,
        'error message missing or empty',
      );
      // kremory's open error path prefixes with "kremory open failed:".
      // Loose check: don't pin exact string (it may evolve).
      return true;
    },
  );
});

// ── 5. Multi-namespace recall (ADR-029c) ────────────────────────────────────

test('multi-namespace recall: in_namespaces returns per-row namespace attribution', async () => {
  const dbPath = tmpDbPath('multi-ns');
  let mem;
  try {
    mem = await JsMemory.open(dbPath);

    await mem.ingest('Alpha team ships kremory v0.1.5.', { namespace: 'team-a' });
    await mem.ingest('Beta team owns the napi binding.', { namespace: 'team-b' });

    const results = await mem.recall('which team ships what?', {
      inNamespaces: ['team-a', 'team-b'],
      k: 10,
    });
    assert.ok(Array.isArray(results));

    // Each row should carry a namespace attribution (ADR-029c).
    // We don't pin which namespaces appear because content extraction is
    // LLM-dependent; we only assert the shape contract: every result with
    // non-null namespace string is one of the queried set.
    for (const r of results) {
      if (r.namespace !== null && r.namespace !== undefined) {
        assert.ok(
          r.namespace === 'team-a' || r.namespace === 'team-b',
          `unexpected namespace attribution: ${r.namespace}`,
        );
      }
    }

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 6. Concurrent ingest (tokio↔Node event loop bridge) ─────────────────────

test('concurrent ingest: 3 parallel ingest calls resolve', async () => {
  const dbPath = tmpDbPath('concurrent');
  let mem;
  try {
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'concurrent' });

    const inputs = [
      'First parallel sentence about apples.',
      'Second parallel sentence about bridges.',
      'Third parallel sentence about clocks.',
    ];

    const results = await Promise.all(
      inputs.map((t) => mem.ingest(t, { namespace: 'concurrent' })),
    );

    assert.equal(results.length, 3, 'expected 3 ingest results');
    for (const r of results) {
      assert.equal(typeof r.episodeEntityId, 'string');
      assert.ok(r.episodeEntityId.length > 0);
    }

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 7. Close idempotency ────────────────────────────────────────────────────

test('close idempotency: second close() does not throw catastrophically', async () => {
  const dbPath = tmpDbPath('close-idem');
  const mem = await JsMemory.open(dbPath);
  try {
    await mem.close();
    // v0.1.0 close() is a no-op; v0.1.1+ adds WAL flush. Second call should
    // either succeed silently or reject with a clean JS Error — NOT panic
    // the process or hang.
    let secondOk = true;
    try {
      await mem.close();
    } catch (err) {
      secondOk = err instanceof Error;
    }
    assert.ok(secondOk, 'second close() crashed or rejected with non-Error');
  } finally {
    cleanup(dbPath);
  }
});

// ── 8. Conflicting namespace selectors (ADR-029c mutual exclusion) ──────────

test('conflicting selectors: namespace + inNamespaces rejects', async () => {
  const dbPath = tmpDbPath('conflict');
  let mem;
  try {
    mem = await JsMemory.open(dbPath);
    await mem.ingest('seed content', { namespace: 'x' });

    await assert.rejects(
      async () => {
        await mem.recall('query', {
          namespace: 'x',
          inNamespaces: ['x', 'y'],
          k: 5,
        });
      },
      (err) => {
        assert.ok(err instanceof Error);
        // kremory's check_selectors emits ConflictingNamespaceSelectors;
        // napi wraps as "kremory recall failed: ...". Loose match.
        return true;
      },
    );

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 9. B1: ingestEpisode with source_id ─────────────────────────────────────

test('B1: ingestEpisode with source_id stores episode and returns result', async () => {
  const dbPath = tmpDbPath('b1-ingest-episode');
  let mem;
  try {
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'b1' });

    const result = await mem.ingestEpisode({
      content: 'first episode about water and rivers.',
      sourceId: 'doc-001',
      namespace: 'b1',
    });

    assert.equal(typeof result.episodeEntityId, 'string', 'episodeEntityId must be string');
    assert.ok(result.episodeEntityId.length > 0, 'episodeEntityId must not be empty');
    assert.ok(!Number.isNaN(Date.parse(result.committedAt)), 'committedAt must be rfc3339');
    assert.ok(Array.isArray(result.warnings), 'warnings must be an array');

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 10. B1: ingestEpisode without source_id ──────────────────────────────────

test('B1: ingestEpisode without source_id works like plain ingest', async () => {
  const dbPath = tmpDbPath('b1-no-source');
  let mem;
  try {
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'b1ns' });

    const result = await mem.ingestEpisode({
      content: 'bare episode without source identity.',
      namespace: 'b1ns',
    });

    assert.equal(typeof result.episodeEntityId, 'string');
    assert.ok(Array.isArray(result.warnings));

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 11. B2/B3: updateMetadata + updateUri ───────────────────────────────────

test('B2/B3: updateMetadata and updateUri succeed after ingestEpisode', async () => {
  const dbPath = tmpDbPath('b2-b3-update');
  let mem;
  try {
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'upd' });

    await mem.ingestEpisode({
      content: 'episode about bridges and spans.',
      sourceId: 'adr-031-binding',
      namespace: 'upd',
    });

    // B3: updateUri
    const uriCount = await mem.updateUri('adr-031-binding', 'docs/adr/031-binding.md');
    assert.ok(uriCount >= 1, `updateUri must update ≥1 rows, got ${uriCount}`);

    // B2: updateMetadata
    const metaCount = await mem.updateMetadata('adr-031-binding', { status: 'active', version: 2 });
    assert.ok(metaCount >= 1, `updateMetadata must update ≥1 rows, got ${metaCount}`);

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 12. B3: updateUri rejects for unknown source_id ─────────────────────────

test('B3: updateUri rejects when source_id not found', async () => {
  const dbPath = tmpDbPath('b3-unknown');
  let mem;
  try {
    mem = await JsMemory.open(dbPath);

    await assert.rejects(
      async () => {
        await mem.updateUri('does-not-exist-xyz-9999', 'path/irrelevant.md');
      },
      (err) => {
        assert.ok(err instanceof Error);
        assert.match(err.message, /kremory updateUri failed/i);
        return true;
      },
    );

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 13. B5: getBySourceId returns episodes ───────────────────────────────────

test('B5: getBySourceId returns episodes matching source_id', async () => {
  const dbPath = tmpDbPath('b5-get');
  let mem;
  try {
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'b5' });

    await mem.ingestEpisode({
      content: 'memo about alpine lakes and glacier melt.',
      sourceId: 'memo-b5-001',
      namespace: 'b5',
    });

    const episodes = await mem.getBySourceId('memo-b5-001', 'b5');
    assert.ok(Array.isArray(episodes), 'getBySourceId must return an array');
    assert.ok(episodes.length >= 1, 'must find at least one episode');

    const ep = episodes[0];
    assert.equal(typeof ep.id, 'number', 'episode.id must be a number');
    assert.equal(ep.sourceId, 'memo-b5-001', 'episode.sourceId must match query arg');
    assert.equal(ep.sourceUri, null, 'episode.sourceUri is always null (known substrate gap)');
    assert.equal(typeof ep.content, 'string', 'episode.content must be string');
    assert.ok(!Number.isNaN(Date.parse(ep.timestamp)), 'episode.timestamp must be rfc3339');

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 14. B5: getBySourceId returns empty array for unknown source_id ──────────

test('B5: getBySourceId returns empty array when source_id not found', async () => {
  const dbPath = tmpDbPath('b5-empty');
  let mem;
  try {
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'b5e' });

    const episodes = await mem.getBySourceId('no-such-source-xyz', 'b5e');
    assert.ok(Array.isArray(episodes));
    assert.equal(episodes.length, 0, 'must return empty array for unknown source_id');

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 15. B6: dream returns JsDreamSummary ─────────────────────────────────────

test('B6: dream returns summary with correct numeric fields', async () => {
  const dbPath = tmpDbPath('b6-dream');
  let mem;
  try {
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'b6' });

    await mem.ingest('dream test: info about wind patterns.', { namespace: 'b6' });

    const summary = await mem.dream({ namespace: 'b6' });

    assert.equal(typeof summary.communitiesUpdated, 'number');
    assert.equal(typeof summary.crossEpisodeMerges, 'number');
    assert.equal(typeof summary.supersessionsRecorded, 'number');
    assert.equal(typeof summary.factsArchived, 'number');
    assert.equal(typeof summary.durationMs, 'number');
    assert.ok(summary.durationMs >= 0, 'durationMs must be non-negative');

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 16. B7: forget removes episode by source_id ──────────────────────────────

test('B7: forget removes episodes by source_id', async () => {
  const dbPath = tmpDbPath('b7-forget');
  let mem;
  try {
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'b7' });

    await mem.ingestEpisode({
      content: 'deletable record about ocean tides.',
      sourceId: 'doc-to-delete-001',
      namespace: 'b7',
    });

    // Verify episode exists.
    const before = await mem.getBySourceId('doc-to-delete-001', 'b7');
    assert.ok(before.length >= 1, 'episode must exist before forget');

    // Forget.
    const deleted = await mem.forget('doc-to-delete-001', 'b7');
    assert.ok(deleted >= 0, 'forget must return a non-negative count');

    // Verify episode gone.
    const after = await mem.getBySourceId('doc-to-delete-001', 'b7');
    assert.equal(after.length, 0, 'episode must be gone after forget');

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 17. B8: reindex always rejects with deferred error ───────────────────────

test('B8: reindex rejects with deferred error', async () => {
  const dbPath = tmpDbPath('b8-reindex');
  let mem;
  try {
    mem = await JsMemory.open(dbPath);

    await assert.rejects(
      async () => {
        await mem.reindex();
      },
      (err) => {
        assert.ok(err instanceof Error);
        assert.match(err.message, /deferred|not yet implemented/i);
        return true;
      },
    );

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 18. B9: recall with filterMetadata ──────────────────────────────────────

test('B9: recall with filterMetadata wires to substrate filter_metadata', async () => {
  const dbPath = tmpDbPath('b9-filter');
  let mem;
  try {
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'b9' });

    // Ingest an episode with metadata.
    await mem.ingestEpisode({
      content: 'design notes about search indexing strategies.',
      sourceId: 'design-b9-001',
      metadata: { docType: 'design', status: 'active' },
      namespace: 'b9',
    });

    // Recall with filterMetadata: substrate validates the key and applies
    // post-filter. Empty results are acceptable (LLM extraction is non-deterministic);
    // we only assert the call succeeds (does not reject).
    const results = await mem.recall('search indexing', {
      namespace: 'b9',
      k: 5,
      filterMetadata: [{ key: 'docType', value: 'design' }],
    });

    assert.ok(Array.isArray(results), 'filtered recall must return array');

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});

// ── 19. B9: recall with invalid filterMetadata key rejects ───────────────────

test('B9: recall with invalid filterMetadata key rejects at await', async () => {
  const dbPath = tmpDbPath('b9-invalid-key');
  let mem;
  try {
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'b9k' });

    // Substrate validates keys at .await time. An empty key must be rejected.
    await mem.ingest('filler episode to have a valid namespace.', { namespace: 'b9k' });

    await assert.rejects(
      async () => {
        await mem.recall('query', {
          namespace: 'b9k',
          filterMetadata: [{ key: '', value: 'x' }],
        });
      },
      (err) => {
        assert.ok(err instanceof Error);
        // kremory surfaces "filter_metadata: key must not be empty"
        assert.match(err.message, /kremory recall failed/i);
        return true;
      },
    );

    await mem.close();
  } finally {
    cleanup(dbPath);
  }
});
