// Minimal smoke: require built binding, open (ollama), one remember.
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../crates/kremory-napi/index.js');

const dbPath = path.join(os.tmpdir(), `kremory-napi-smoke-${Date.now()}-${process.pid}.db`);

async function main() {
  console.log('[smoke] opening Memory (ollama auto → gemma4:e4b)…');
  const mem = await Memory.open(dbPath, { defaultNamespace: 'smoke' });
  assert.ok(mem, 'Memory.open returned falsy');
  console.log('[smoke] open OK');

  const t0 = Date.now();
  const r = await mem.remember({
    content: 'Ada Lovelace collaborated with Charles Babbage on the Analytical Engine.',
    namespace: 'smoke',
  });
  console.log(`[smoke] remember OK in ${Date.now() - t0}ms → episodeEntityId=${r.episodeEntityId}`);
  assert.equal(typeof r.episodeEntityId, 'string');
  assert.ok(r.episodeEntityId.length > 0);
  assert.ok(Array.isArray(r.warnings));

  await mem.close();
  console.log('[smoke] PASS');
}

main().catch((e) => {
  console.error('[smoke] FAIL:', e);
  process.exit(1);
}).finally(() => {
  try { fs.unlinkSync(dbPath); } catch {}
});
