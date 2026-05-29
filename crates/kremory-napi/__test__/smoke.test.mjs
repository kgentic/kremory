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
