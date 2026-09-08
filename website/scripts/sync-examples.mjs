// Generate website/docs/examples/*.md from the crate's runnable examples.
//
// The .rs file is the source of truth: it is compiled by `cargo test` and RUN by a
// human, so it cannot quietly stop working. A hand-copied duplicate in markdown
// would go stale the moment the API moved and nobody would notice — which is the
// exact failure this repo already avoids for CHANGELOG.md via sync-changelog.mjs.
// Same shape here: derive the page, gitignore the output, run from `prebuild`.
//
// Deliberately NOT a general Rust-to-markdown transformer. It does two things —
// lift the `//!` module doc into prose, and fence the rest as Rust — and asserts
// its input rather than guessing at it.

import {readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const EXAMPLES_DIR = resolve(HERE, '../../crates/kremory/examples');
const OUT_DIR = resolve(HERE, '../docs/examples');

// Explicit allowlist, in reading order. Adding an example to the site is a
// deliberate act — the crate also holds internal diagnostics (alias_probe,
// graph_health) that are NOT teaching material and must never be published here.
const PUBLISHED = [
  {
    file: 'offline_remember_recall.rs',
    slug: 'offline-remember-recall',
    title: 'Save something, get it back',
    position: 1,
  },
  {
    file: 'remembers_across_sessions.rs',
    slug: 'remembers-across-sessions',
    title: 'Remembering when things changed',
    position: 2,
  },
  {
    file: 'agent_memory_with_ollama.rs',
    slug: 'agent-memory-with-ollama',
    title: 'Building a graph from prose (needs Ollama)',
    position: 3,
  },
];

mkdirSync(OUT_DIR, {recursive: true});

for (const {file, slug, title, position} of PUBLISHED) {
  const src = resolve(EXAMPLES_DIR, file);

  let raw;
  try {
    raw = readFileSync(src, 'utf8');
  } catch (err) {
    throw new Error(
      `Cannot read ${src} — an example listed in PUBLISHED does not exist. ` +
        `Refusing to publish a page for a file that is not there. (${err.code})`,
    );
  }

  // Precondition: fail loudly if the source stops looking like what we expect,
  // rather than emitting a plausible but wrong page.
  if (!raw.startsWith('//!')) {
    throw new Error(
      `${src} does not begin with a \`//!\` module doc comment — refusing to ` +
        `generate a page with no explanation from an unrecognised file shape.`,
    );
  }

  const lines = raw.split('\n');
  const docLines = [];
  let i = 0;
  for (; i < lines.length; i++) {
    const l = lines[i];
    if (l.startsWith('//!')) {
      // Strip `//!` plus the single conventional space, but no more — deeper
      // indentation is meaningful markdown (nested lists, fenced blocks).
      docLines.push(l.replace(/^\/\/! ?/, ''));
      continue;
    }
    if (l.trim() === '') continue; // blank lines inside the header block
    break;
  }
  const code = lines.slice(i).join('\n').trim();

  if (code.length === 0) {
    throw new Error(`${src} has a module doc but no code beneath it.`);
  }

  const page = `---
title: ${JSON.stringify(title)}
sidebar_position: ${position}
---

{/* GENERATED FILE — do not edit.
    Source: crates/kremory/examples/${file}
    Regenerate: npm run sync:examples (runs automatically on prebuild). */}

${docLines.join('\n').trim()}

## The whole program

Run it with \`cargo run --example ${file.replace(/\.rs$/, '')}\`.

\`\`\`rust
${code}
\`\`\`
`;

  writeFileSync(resolve(OUT_DIR, `${slug}.md`), page);
  console.log(`sync-examples: ${file} -> docs/examples/${slug}.md`);
}
