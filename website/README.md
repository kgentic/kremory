# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

Content under `docs/` is migrated from this repo's canonical `README.md` and `docs/*.md`
(those files remain the source of truth and are NOT deleted — see
`docs/specs/public-docs-and-api-surface-audit/spec.md` RULE-009/RULE-010). This site is
scaffolded and build-verified locally only; it is **not deployed** (deploy is a separate,
maintainer-gated phase — repo-public + DNS decisions are pending).

## Theme & brand assets

The site's visual identity implements the **"Architectural Systems Precision"** design system.
The full system (colour roles, type scale, spacing, elevation and component specs) is recorded
at `.context/stitch/kremory-redesign/DESIGN.md`, alongside the source mock it came from.

Everything theme-related lives in three places:

| Where | What |
|---|---|
| `src/css/custom.css` | The whole theme — Infima variable overrides plus the rules that enforce the design system's three load-bearing constraints (planar depth, milled corners, carbon code surfaces). Each block cites the design-system section it implements. |
| `src/pages/` + `src/components/HomepageFeatures/` | Homepage hero and feature panels. Copy is unchanged from the scaffold; only presentation was rewritten. |
| `static/img/` | Derived brand assets. |

**Brand master:** `static/brand/kremory-logo.png` (1254x1254, transparent). Everything in
`static/img/` is derived from it and can be regenerated:

- `logo.png` / `logo-dark.png` — cropped to content and squared. The dark variant recolours
  only the charcoal strokes to bone so the mark stays legible on the carbon page; the gold leg
  and junction node are untouched.
- `favicon-64.png`, `apple-touch-icon.png` — downscales of `logo.png`.
- `social-card.png` — 1280x640 Open Graph card.

Two things worth knowing before regenerating:

1. **The social card's type is set in system Helvetica, not Inter.** Inter and JetBrains Mono
   are installed as `@fontsource` web fonts (woff/woff2), which fontconfig cannot use, so the
   rasteriser silently falls back. If those fonts are ever installed system-wide, the card
   should be regenerated to pick them up.
2. **No metric appears anywhere in this theme, by design.** The source mock filled its panels
   with invented latency, footprint, star-count and competitor-benchmark figures. None were
   measured, so none were carried over — see the audit table in
   `.context/stitch/kremory-redesign/README.md`.

## Installation

```bash
npm install
```

**Note**: feel free to use the package manager of your choice.

## Local Development

```bash
npm run start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true npm run deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> npm run deploy
```

If you are using GitHub Pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
