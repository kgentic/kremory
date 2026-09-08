import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Manual sidebar. Every doc migrated from the repo's `docs/` corpus + `README.md`
 * must appear here (RULE-009: 100% of existing docs reachable from nav).
 *
 * The API reference is a category rather than a single entry: it was one
 * 1,501-line page until 2026-09-08, which made every cross-reference into it a
 * scroll-and-hunt. Its landing page is `api/index`, wired as the category link
 * so clicking the category header goes somewhere useful rather than just
 * toggling.
 *
 * Page order inside the category is adoption order — open a Memory, write,
 * read, maintain the graph, integrate — not alphabetical.
 */
const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    'getting-started',
    {
      type: 'category',
      label: 'Examples',
      collapsed: false,
      // GENERATED pages — `website/scripts/sync-examples.mjs` derives these from
      // the runnable `.rs` files under `crates/kremory/examples/` at prebuild, and
      // `docs/examples/` is gitignored. Editing the markdown is a mistake; edit the
      // example. Ordered by reading order (offline basics first), not alphabetically.
      items: [
        'examples/offline-remember-recall',
        'examples/remembers-across-sessions',
        'examples/agent-memory-with-ollama',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      collapsed: false,
      link: {type: 'doc', id: 'api/index'},
      items: [
        'api/setup',
        'api/namespaces',
        'api/ingest',
        'api/recall',
        'api/bi-temporal',
        'api/dream',
        'api/reversibility',
        'api/async-and-events',
        'api/advanced',
        'api/feature-flags',
        'api/node-binding',
      ],
    },
    {
      type: 'category',
      label: 'Releases',
      collapsed: false,
      // `changelog` is GENERATED at build time from crates/kremory/CHANGELOG.md
      // (see scripts/sync-changelog.mjs, wired into prebuild). It is gitignored:
      // the crate file is release-managed, and a committed copy would drift the
      // moment a release lands.
      items: ['releases/upgrade-guide', 'changelog'],
    },
    {
      type: 'category',
      label: 'Operations',
      collapsed: false,
      items: ['observability', 'error-handling-policy'],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: false,
      items: ['comparison', 'benchmarks', 'eval', 'eval-fixtures', 'testing'],
    },
  ],
};

export default sidebars;
