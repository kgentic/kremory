import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Manual sidebar — mirrors the plan's "Getting Started / API Reference /
 * Operations / Reference" categorization (plan.md Phase 5). Every doc
 * migrated from the repo's `docs/*.md` + `README.md` corpus must appear here
 * (RULE-009: 100% of existing docs reachable from nav).
 */
const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    'getting-started',
    {
      type: 'category',
      label: 'API Reference',
      collapsed: false,
      items: ['api'],
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
