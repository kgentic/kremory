// Smoke-runs every executable example in this directory and fails loud on
// the first regression, instead of letting them rot unexecuted between
// manual re-reads (see examples/README.md's own findings history — several
// examples asserted a bug that had since been fixed, and nothing caught the
// drift because nothing ran them).
//
// `10-unmerge.ts` is intentionally excluded — per its own header comment and
// examples/README.md, it is type-check-only (`pnpm test:types` covers it;
// triggering a real merge is not deterministically reproducible).
//
// REQUIRES: native module built (`pnpm build:debug`); a chat provider env
// var set (OLLAMA_HOST, OPENAI_API_KEY, or ANTHROPIC_API_KEY) — see
// examples/README.md "Prerequisites".
//
// Run: `node examples/run-all.mjs`

import { spawn } from 'node:child_process';
import { readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

const EXCLUDE = new Set(['run-all.mjs']);
const TIMEOUT_MS = {
  // Dream runs multiple LLM-backed consolidation passes; give it more room
  // than the flat default below.
  '05-dream-and-consolidation.mjs': 180_000,
};
const DEFAULT_TIMEOUT_MS = 90_000;

const files = readdirSync(here)
  .filter((f) => f.endsWith('.mjs') && !EXCLUDE.has(f))
  .sort();

function runOne(file) {
  return new Promise((resolve) => {
    const timeoutMs = TIMEOUT_MS[file] ?? DEFAULT_TIMEOUT_MS;
    const child = spawn(process.execPath, [path.join(here, file)], {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: process.env,
    });
    let out = '';
    child.stdout.on('data', (d) => (out += d));
    child.stderr.on('data', (d) => (out += d));
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      resolve({ file, ok: false, out, reason: `timed out after ${timeoutMs}ms` });
    }, timeoutMs);
    child.on('exit', (code) => {
      clearTimeout(timer);
      resolve({ file, ok: code === 0, out, reason: code === 0 ? null : `exit code ${code}` });
    });
  });
}

const results = [];
for (const file of files) {
  process.stdout.write(`▶ ${file} ... `);
  const result = await runOne(file);
  results.push(result);
  console.log(result.ok ? 'PASS' : `FAIL (${result.reason})`);
  if (!result.ok) {
    console.log(`--- ${file} output (last 40 lines) ---`);
    console.log(result.out.split('\n').slice(-40).join('\n'));
  }
}

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} examples passed`);
if (failed.length > 0) {
  console.log(`FAILED: ${failed.map((r) => r.file).join(', ')}`);
  process.exit(1);
}
