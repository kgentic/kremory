/**
 * kremory-napi smoke test — full round-trip from Node.js.
 *
 * REQUIRES: `pnpm install && pnpm build` (or `npm install && npm run build`)
 * in `crates/kremory-napi/` first. The native .node binary must be present.
 *
 * NOT run in CI yet. Add to CI matrix once napi-rs prebuilt binaries are
 * configured per ADR-030 §6 open item 7.
 *
 * Prerequisite: OLLAMA_HOST or OPENAI_API_KEY or ANTHROPIC_API_KEY in env
 * for the env-auto provider detection path.
 */

const { JsMemory } = require('../index.js');
const os = require('os');
const path = require('path');
const fs = require('fs');

async function runSmoke() {
  const dbPath = path.join(os.tmpdir(), `kremory-smoke-${Date.now()}.db`);
  console.log(`[smoke] Using temp DB: ${dbPath}`);

  let mem;
  try {
    // Phase 1: open in-memory (temp file)
    console.log('[smoke] Opening Memory...');
    mem = await JsMemory.open(dbPath, { defaultNamespace: 'smoke-test' });
    console.log('[smoke] Memory opened OK');

    // Phase 2: ingest a single sentence
    const ingestResult = await mem.ingest(
      'James prefers concise, data-driven technical decisions with minimal ceremony.',
      { namespace: 'smoke-test', contentType: 'chat' }
    );
    console.log('[smoke] Ingest OK:', JSON.stringify(ingestResult));

    // Phase 3: recall by partial match
    const results = await mem.recall('what does James prefer?', {
      namespace: 'smoke-test',
      k: 5,
    });
    console.log(`[smoke] Recall OK: ${results.length} result(s)`);
    if (results.length > 0) {
      const first = results[0];
      console.log('[smoke] Top result:', JSON.stringify({
        entity_id: first.entityId,
        entity_name: first.entityName,
        score: first.score,
        incomplete: first.incomplete,
      }));
    }

    // Phase 4: close
    await mem.close();
    console.log('[smoke] Close OK');

    console.log('[smoke] PASS');
  } catch (err) {
    console.error('[smoke] FAIL:', err);
    process.exitCode = 1;
  } finally {
    // Clean up temp file
    try {
      if (fs.existsSync(dbPath)) {
        fs.unlinkSync(dbPath);
      }
    } catch (_) {
      // best-effort cleanup
    }
  }
}

runSmoke();
