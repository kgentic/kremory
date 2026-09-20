import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'kremory',
  tagline: 'The SQLite of agent memory',
  favicon: 'img/favicon-64.png',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Deploy target, set 2026-09-20 when the maintainer-gated deploy phase the
  // previous note deferred to actually happened.
  //
  // PROJECT SITE, not a custom domain: `kremory.dev` is unregistered (verified
  // — whois returns only an IANA stub, and there is no A record), so the
  // previous `url: kremory.dev` / `baseUrl: '/'` pair could not have worked.
  // A project site is served from `<org>.github.io/<projectName>/`, so baseUrl
  // MUST carry the repo path — leaving it '/' is the classic Docusaurus
  // misconfiguration where every asset 404s under a path-prefixed host.
  //
  // To move to a custom domain later: register it, set `url` to it, set
  // `baseUrl` back to '/', and add `website/static/CNAME` containing the bare
  // hostname. All three, or the site breaks in a different direction.
  url: 'https://kgentic.github.io',
  baseUrl: '/kremory/',

  // GitHub pages deployment config (only consumed by `docusaurus deploy`,
  // which this spec does not run — kept accurate for when deploy does happen).
  organizationName: 'kgentic',
  projectName: 'kremory',

  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  headTags: [
    {
      tagName: 'link',
      attributes: {
        rel: 'apple-touch-icon',
        href: '/img/apple-touch-icon.png',
      },
    },
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/kgentic/kremory/tree/main/website/',
        },
        // No blog for this project's docs site — docs-only.
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.png',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'kremory',
      logo: {
        alt: 'kremory logo',
        src: 'img/logo.png',
        // The mark's charcoal strokes disappear against the carbon page in
        // dark mode; the dark variant recolours only those strokes to bone
        // and leaves the gold leg and junction node untouched.
        srcDark: 'img/logo-dark.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://github.com/kgentic/kremory',
          label: 'GitHub',
          position: 'right',
        },
        {
          href: 'https://crates.io/crates/kremory',
          label: 'crates.io',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/getting-started',
            },
            {
              label: 'API Reference',
              to: '/docs/api',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/kgentic/kremory',
            },
            {
              label: 'crates.io',
              href: 'https://crates.io/crates/kremory',
            },
            {
              label: 'docs.rs',
              href: 'https://docs.rs/kremory',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} kgentic. Apache-2.0 licensed.`,
    },
    prism: {
      // Fenced code renders on the same carbon surface in BOTH colour modes
      // (design system rule 3 — a terminal does not turn white because the
      // page did), so both slots take a dark base theme. The actual token
      // palette is overridden in src/css/custom.css.
      theme: prismThemes.vsDark,
      darkTheme: prismThemes.vsDark,
      additionalLanguages: ['rust', 'toml', 'bash', 'json'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
