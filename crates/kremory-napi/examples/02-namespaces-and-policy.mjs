// kremory-napi namespaces + policy — defaultNamespace, per-call namespace
// override, registerNamespace, upgradeNamespacePolicy (Mutable -> AppendOnly
// one-way ratchet), and multi-namespace recall (inNamespaces + bestEffort +
// perNamespaceTopK).
//
// Uses `skipExtraction: true` throughout — this file is about namespace
// plumbing, not extraction quality, so it runs fast and deterministically.
// A chat provider still needs to be configured at `Memory.open()` time even
// though no call here triggers real extraction.
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED (v0.4.1) — this imports the
// LOCALLY BUILT native module, not a published npm package.
//
// REQUIRES: native module built (`pnpm build:debug`);
//   OLLAMA_HOST=http://localhost:11434 (or OPENAI_API_KEY / ANTHROPIC_API_KEY)
//   set even though extraction itself is skipped.
//
// Run: `node examples/02-namespaces-and-policy.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

const dbPath = path.join(os.tmpdir(), `kremory-example-namespaces-${Date.now()}.db`);

console.log('[02] opening Memory (no defaultNamespace set)');
const mem = await Memory.open(dbPath);

try {
  // ── registerNamespace: explicit registration ahead of any writes ────────
  console.log('[02] registerNamespace("team-a"), registerNamespace("team-b")');
  await mem.registerNamespace('team-a');
  await mem.registerNamespace('team-b');
  // Idempotent for the same policy — calling again must not throw.
  await mem.registerNamespace('team-a');

  // ── upgradeNamespacePolicy: Mutable -> AppendOnly one-way ratchet ────────
  console.log('[02] upgradeNamespacePolicy("team-a") — Mutable -> AppendOnly');
  await mem.upgradeNamespacePolicy('team-a');
  // Idempotent when already AppendOnly.
  await mem.upgradeNamespacePolicy('team-a');

  // AppendOnly rejects forget() in that namespace (documented on `forget`).
  await mem.remember({
    content: 'Alpha team ships kremory v0.1.5.',
    namespace: 'team-a',
    skipExtraction: true,
  });
  await assert.rejects(
    () => mem.forget('does-not-matter', 'team-a'),
    (err) => {
      assert.ok(err instanceof Error);
      console.log('[02] forget() on AppendOnly namespace rejected as documented:', err.message);
      return true;
    },
  );

  // ── per-call namespace override (no defaultNamespace on the handle) ─────
  console.log('[02] remember() with explicit per-call namespace "team-b"');
  await mem.remember({
    content: 'Beta team owns the napi binding.',
    namespace: 'team-b',
    skipExtraction: true,
  });

  // No namespace anywhere -> rejects with a namespace-required error.
  await assert.rejects(
    () => mem.remember({ content: 'orphan, no namespace anywhere', skipExtraction: true }),
    (err) => {
      assert.match(err.message, /namespace required/i);
      return true;
    },
  );

  // ── multi-namespace recall: inNamespaces + bestEffort + perNamespaceTopK ─
  console.log('[02] recall() across ["team-a", "team-b"] with bestEffort + perNamespaceTopK');
  const results = await mem.recall('which team owns what?', {
    inNamespaces: ['team-a', 'team-b'],
    bestEffort: true,
    perNamespaceTopK: 5,
    k: 10,
  });
  assert.ok(Array.isArray(results), 'multi-namespace recall must return an array');
  console.log(`[02] multi-namespace recall returned ${results.length} result(s)`);
  for (const r of results) {
    if (r.namespace != null) {
      assert.ok(
        r.namespace === 'team-a' || r.namespace === 'team-b',
        `unexpected namespace attribution: ${r.namespace}`,
      );
    }
    console.log(`  - entityId=${r.entityId} namespace=${r.namespace}`);
  }

  // `namespace` + `inNamespaces` together is a documented conflict — rejects
  // at `.await` time.
  await assert.rejects(
    () =>
      mem.recall('q', {
        namespace: 'team-a',
        inNamespaces: ['team-a', 'team-b'],
      }),
    (err) => {
      assert.ok(err instanceof Error);
      console.log('[02] conflicting namespace selectors rejected as documented:', err.message);
      return true;
    },
  );

  console.log('[02] PASS');
} finally {
  await mem.close();
  try { fs.unlinkSync(dbPath); } catch { /* best-effort cleanup */ }
}
