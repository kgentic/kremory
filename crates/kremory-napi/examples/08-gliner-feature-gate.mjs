// kremory-napi `opts.gliner` — GLiNER extraction config (ADR-039 Part 10).
//
// This locally built native module (`../kremory.<platform>.node`) was built
// with DEFAULT features only (`content-search`) — it does NOT include
// `ner`. `index.d.ts`'s own doc comment on `GlinerConfig` documents exactly
// this case: "Requires kremory-napi built with `--features ner`. Passing
// this field on a non-ner build causes `Memory.open` to return an error."
//
// This example deliberately exercises THAT documented rejection path — it
// is a real, runnable proof that the documented feature-gate behaviour is
// correct on this build, not a demonstration of live GLiNER extraction
// (which would require rebuilding the native module with `--features ner`,
// a Rust-side build change out of scope for this example set).
//
// ⚠️ `@kgentic-ai/kremory-node` is UNPUBLISHED (v0.4.1) — this imports the
// LOCALLY BUILT native module, not a published npm package.
//
// REQUIRES: native module built (`pnpm build:debug`, default features);
//   OLLAMA_HOST=http://localhost:11434 (or OPENAI_API_KEY / ANTHROPIC_API_KEY)
//   — gliner requires `llm` to also be set per index.d.ts's own doc comment.
//
// Run: `node examples/08-gliner-feature-gate.mjs`

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

const dbPath = path.join(os.tmpdir(), `kremory-example-gliner-gate-${Date.now()}.db`);

console.log('[08] Memory.open() with opts.gliner={} on a build WITHOUT --features ner');
await assert.rejects(
  () => Memory.open(dbPath, { gliner: {}, defaultNamespace: 'gliner-demo' }),
  (err) => {
    assert.ok(err instanceof Error);
    console.log('[08] rejected as documented:', err.message);
    assert.match(
      err.message,
      /ner|FeatureDisabled|feature/i,
      `expected a feature-gate error mentioning "ner"/"FeatureDisabled"/"feature"; got: ${err.message}`,
    );
    return true;
  },
);

// GlinerConfig's fields (modelPath, threshold) are documented as RESERVED /
// inert on the wire today — accepted but not threaded through even on a
// ner-enabled build. Confirm the SAME feature-gate rejection fires
// regardless of which reserved fields are populated (shape acceptance is
// not what's being gated — the `ner` cargo feature is).
console.log('[08] Memory.open() with opts.gliner={ modelPath, threshold } — same feature gate');
await assert.rejects(
  () =>
    Memory.open(dbPath, {
      gliner: { modelPath: '/tmp/does-not-matter.onnx', threshold: 0.5 },
      defaultNamespace: 'gliner-demo',
    }),
  (err) => {
    assert.ok(err instanceof Error);
    console.log('[08] rejected as documented (reserved fields present):', err.message);
    return true;
  },
);

try { fs.unlinkSync(dbPath); } catch { /* best-effort cleanup; open() never created the file since it rejected before persisting */ }

console.log('[08] PASS');
