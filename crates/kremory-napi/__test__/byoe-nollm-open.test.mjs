// kremory-napi BYOE "NoLlm typestate" regression suite (node:test runner).
//
// F46 (public-docs-and-api-surface-audit phase1-findings.md): `Memory.open`
// with ONLY `{ withEmbedder, extractor }` set (no `llm` knob — there isn't
// even one on `OpenOptions`) used to reject with "no LLM provider
// configured" unless OLLAMA_HOST/OPENAI_API_KEY/ANTHROPIC_API_KEY happened
// to be set — directly contradicting the documented "NoLlm typestate"
// (`ExtractorKind::Custom`). Root cause: `open_with_js_embedder` called
// `bridge::resolve_env_llm()` UNCONDITIONALLY before ever checking whether a
// BYOE `extractor` was supplied.
//
// A first draft of the fix unconditionally SKIPPED LLM detection whenever an
// extractor was present — which regressed the substrate's own
// `memory_builder_compat_matrix.rs::row5_llm_and_custom_extractor_builds_memory`
// ("LLM + custom extractor → Ok(Memory), custom wins"): an available LLM
// must still be wired for entity RESOLUTION (`ingest_with`'s
// `CascadeResolver`), independent of which extractor produced the entities.
// This file exercises BOTH shapes against the REAL native module + a real
// Ollama instance (per this repo's own `EMBED_DIM`/env-var conventions in
// `smoke-embedder.test.mjs`), not a stub.
//
// REQUIRES: `pnpm build` (or `npm run build`) in `crates/kremory-napi/` first.
// The native .node binary must be present at index.js. OLLAMA_HOST (or
// OPENAI_API_KEY / ANTHROPIC_API_KEY) must be reachable for test 2/3 below —
// test 1 deliberately clears all three to prove the NoLlm path.
//
// Run: `OLLAMA_HOST=http://localhost:11434 node --test __test__/byoe-nollm-open.test.mjs`

import { test } from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');

// 384, NOT an arbitrary custom dim: F47 (a separate, un-fixed finding) means
// a BYOE extractor's real entities fail to embed at any dim other than 384
// ("SQLite failure: vector index(insert): dimensions are different"),
// independent of this file's F46 concern. Using 384 keeps this file scoped
// to F46 only; see F47 in phase1-findings.md for the dimension bug itself.
const EMBED_DIM = 384;
const LLM_ENV_VARS = ['OLLAMA_HOST', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'];

function tmpDbPath(tag) {
  return path.join(os.tmpdir(), `kremory-byoe-nollm-${tag}-${Date.now()}-${process.pid}.db`);
}

function cleanup(p) {
  try {
    if (fs.existsSync(p)) fs.unlinkSync(p);
  } catch { /* best-effort cleanup */ }
}

function makeEmbedder(dim) {
  return async function embedText(text) {
    const safeText = text == null ? '' : String(text);
    const vec = new Array(dim).fill(0);
    for (let i = 0; i < safeText.length; i++) {
      vec[i % dim] += safeText.charCodeAt(i) / 255.0;
    }
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
    return vec.map((v) => v / norm);
  };
}

function makeNoopExtractor() {
  return {
    name: 'noop-extractor',
    // Real convention (F45): called with (errSlot, text) — errSlot always null.
    extract: async (_errSlot, _text) => ({ entities: [], facts: [] }),
  };
}

test('Memory.open({ withEmbedder, extractor }) succeeds with ZERO LLM env vars set', async () => {
  const saved = Object.fromEntries(LLM_ENV_VARS.map((k) => [k, process.env[k]]));
  for (const k of LLM_ENV_VARS) delete process.env[k];

  const dbPath = tmpDbPath('open-no-llm');
  let mem;
  try {
    mem = await Memory.open(dbPath, {
      withEmbedder: makeEmbedder(EMBED_DIM),
      embeddingDim: EMBED_DIM,
      extractor: makeNoopExtractor(),
    });
    assert.ok(mem, 'Memory.open must succeed with a BYOE extractor and no LLM env var at all');
  } finally {
    for (const k of LLM_ENV_VARS) {
      if (saved[k] !== undefined) process.env[k] = saved[k];
    }
    cleanup(dbPath);
  }
});

test('Memory.open({ withEmbedder, extractor }) + remember() still needs an LLM for entity resolution (documented residual gap, not this finding)', async () => {
  const saved = Object.fromEntries(LLM_ENV_VARS.map((k) => [k, process.env[k]]));
  for (const k of LLM_ENV_VARS) delete process.env[k];

  const dbPath = tmpDbPath('remember-no-llm');
  let mem;
  try {
    mem = await Memory.open(dbPath, {
      withEmbedder: makeEmbedder(EMBED_DIM),
      embeddingDim: EMBED_DIM,
      extractor: {
        name: 'entity-extractor',
        extract: async (_errSlot, _text) => ({
          entities: [{ name: 'Alice', label: 'Person' }],
          facts: [],
        }),
      },
    });

    await assert.rejects(
      mem.remember({ content: 'Alice is here.', namespace: 'ns1' }),
      /requires an LLM provider/,
      'entity resolution still requires an LLM even via a BYOE extractor — this is NOT F46 (open-time), it is a separate, un-fixed substrate behaviour',
    );
  } finally {
    for (const k of LLM_ENV_VARS) {
      if (saved[k] !== undefined) process.env[k] = saved[k];
    }
    cleanup(dbPath);
  }
});

test('Memory.open({ withEmbedder, extractor }) + a REAL configured LLM still wires it (compat-matrix Row 5 — custom extractor wins for extraction, LLM stays available for resolution)', async () => {
  const hasLlm =
    process.env.OLLAMA_HOST || process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY;
  if (!hasLlm) {
    // Skip rather than fail — this test needs a real reachable LLM, unlike
    // the two above which deliberately test the NO-llm path.
    return;
  }

  const dbPath = tmpDbPath('row5-llm-plus-extractor');
  let mem;
  try {
    mem = await Memory.open(dbPath, {
      withEmbedder: makeEmbedder(EMBED_DIM),
      embeddingDim: EMBED_DIM,
      extractor: {
        name: 'entity-extractor',
        extract: async (_errSlot, text) => ({
          entities: [{ name: 'Alice', label: 'Person' }],
          facts: [],
        }),
      },
    });

    const result = await mem.remember({ content: 'Alice is here.', namespace: 'ns1' });
    assert.ok(result, 'remember() must succeed when a real LLM IS configured alongside the extractor');
  } finally {
    if (mem) await mem.close();
    cleanup(dbPath);
  }
});
