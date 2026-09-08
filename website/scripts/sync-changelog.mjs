// Generate website/docs/changelog.md from the crate's own CHANGELOG.
//
// The crate file is the source of truth: it ships to crates.io and is managed
// by release-please. A hand-copied duplicate would go stale the moment a
// release landed and nobody would notice, so this derives the page instead —
// run from `prebuild` and `prestart`, and the output is gitignored.
//
// Deliberately NOT a general markdown transformer. It does the two things the
// site needs (frontmatter, and demoting the file's H1 so Docusaurus renders one
// title rather than two) and asserts its input rather than guessing at it.

import {readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(HERE, '../../crates/kremory/CHANGELOG.md');
const OUT = resolve(HERE, '../docs/changelog.md');

const raw = readFileSync(SRC, 'utf8');

// Precondition: fail loudly if the source stops looking like what we expect,
// rather than emitting a plausible but wrong page.
if (!raw.startsWith('# Changelog')) {
  throw new Error(
    `${SRC} does not start with "# Changelog" — refusing to generate a page ` +
      `from an unrecognised file rather than guess at its shape.`,
  );
}
const releaseHeadings = (raw.match(/^## \[/gm) ?? []).length;
if (releaseHeadings < 1) {
  throw new Error(
    `${SRC} contains no "## [version]" headings — it is either empty or its ` +
      `format changed. Refusing to publish an empty changelog.`,
  );
}

const body = raw.replace(/^# Changelog\n/, '').trimStart();

const page = `---
title: Changelog
description: Release history for the kremory crate.
---

{/* GENERATED FILE — do not edit.
    Source: crates/kremory/CHANGELOG.md
    Regenerate: npm run sync:changelog (runs automatically on build and start) */}

# Changelog

Release history for the \`kremory\` crate, mirrored from
[\`crates/kremory/CHANGELOG.md\`](https://github.com/kgentic/kremory/blob/main/crates/kremory/CHANGELOG.md).
Upgrading across a breaking change is covered in the
[upgrade guide](./releases/upgrade-guide.md).

${body}
`;

mkdirSync(dirname(OUT), {recursive: true});
writeFileSync(OUT, page);

// Postcondition: verify what landed, rather than reporting from intent.
const written = readFileSync(OUT, 'utf8');
const writtenReleases = (written.match(/^## \[/gm) ?? []).length;
if (writtenReleases !== releaseHeadings) {
  throw new Error(
    `wrote ${writtenReleases} release sections but the source has ` +
      `${releaseHeadings} — the transform dropped content.`,
  );
}
console.log(
  `sync-changelog: wrote docs/changelog.md (${writtenReleases} releases)`,
);
