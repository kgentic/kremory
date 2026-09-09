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
    file: 'multi_tenant_isolation.rs',
    slug: 'multi-tenant-isolation',
    title: 'One database, many customers',
    position: 9,
  },
  {
    file: 'correcting_the_record.rs',
    slug: 'correcting-the-record',
    title: 'Correcting the record',
    position: 3,
  },
  {
    file: 'searching_documents.rs',
    slug: 'searching-documents',
    title: 'Searching documents, not building a graph',
    position: 5,
  },
  {
    file: 'bulk_import.rs',
    slug: 'bulk-import',
    title: 'Importing data you already have',
    position: 7,
  },
  {
    file: 'gdpr_erasure_by_source.rs',
    slug: 'gdpr-erasure-by-source',
    title: 'Right to erasure, scoped to a source',
    position: 11,
  },
  {
    file: 'ingest_without_blocking.rs',
    slug: 'ingest-without-blocking',
    title: 'Ingesting without blocking your handler',
    position: 8,
  },
  {
    file: 'deleting_and_restoring.rs',
    slug: 'deleting-and-restoring',
    title: 'Deleting one fact, and putting it back',
    position: 13,
  },
  {
    file: 'changing_embedding_model.rs',
    slug: 'changing-embedding-model',
    title: 'Changing embedding model without re-ingesting',
    position: 15,
  },
  {
    file: 'undoing_a_correction.rs',
    slug: 'undoing-a-correction',
    title: 'When the correction itself was wrong',
    position: 4,
  },
  {
    file: 'domain_entity_types.rs',
    slug: 'domain-entity-types',
    title: 'Teaching it your domain vocabulary',
    position: 10,
  },
  {
    file: 'dream_on_a_schedule.rs',
    slug: 'dream-on-a-schedule',
    title: 'Letting consolidation run on its own',
    position: 16,
  },
  {
    file: 'hosted_providers.rs',
    slug: 'hosted-providers',
    title: 'One line to a hosted model, and its trade-offs',
    position: 19,
  },
  {
    file: 'agent_memory_with_ollama.rs',
    slug: 'agent-memory-with-ollama',
    title: 'Building a graph from prose (needs Ollama)',
    position: 18,
  },
  {
    file: 'undoing_a_bad_change.rs',
    slug: 'undoing-a-bad-change',
    title: 'Undoing a change that was wrong',
    position: 14,
  },
  {
    file: 'append_only_namespace.rs',
    slug: 'append-only-namespace',
    title: 'A namespace that refuses to be rewritten',
    position: 12,
  },
  {
    file: 'a_long_document.rs',
    slug: 'a-long-document',
    title: 'A document longer than the embedding window',
    position: 6,
  },
  {
    file: 'two_handles_one_database.rs',
    slug: 'two-handles-one-database',
    title: 'Two processes, one database',
    position: 17,
  },
];

mkdirSync(OUT_DIR, {recursive: true});

// ── Index page, single-sourced from the crate's own examples/README.md ───────
//
// `docs/examples/` is gitignored because it is generated, so a hand-written
// index here would be lost. Generating it from the README keeps ONE copy of the
// routing table — the one a reader of the repo sees — instead of two that drift.
//
// Bare `example_name` references in the README become links to the generated
// pages, so the same table works in both places.
{
  const readme = readFileSync(
    resolve(HERE, '../../crates/kremory/examples/README.md'),
    'utf8',
  );

  const slugFor = new Map(
    PUBLISHED.map((e) => [e.file.replace(/\.rs$/, ''), e.slug]),
  );

  const linked = readme
    // `example_name` -> [`example_name`](./slug)
    .replace(/`([a-z0-9_]+)`/g, (whole, name) =>
      slugFor.has(name) ? `[\`${name}\`](./${slugFor.get(name)})` : whole,
    )
    // the repo-relative run command is meaningless on the site
    .replace(/^# Examples — start here$/m, '# Examples');

  const indexPage = `---
sidebar_position: 0
title: Examples
description: Fifteen runnable programs, single-sourced from the crate.
---

${linked}
`;

  writeFileSync(resolve(OUT_DIR, 'index.md'), indexPage);
  console.log('sync-examples: examples/README.md -> docs/examples/index.md');
}

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
