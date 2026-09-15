# Security Policy

## Supported versions

kremory is pre-1.0. There is one supported line: **whatever is the latest published `0.x`
release on [crates.io](https://crates.io/crates/kremory)**. Security fixes land on `main` and
ship in the next patch/minor release — there is no backport policy for older `0.x` releases
while the project is pre-1.0. (Deliberately not naming a specific version number here — pre-1.0,
that number changes often enough that a hardcoded one goes stale faster than this file gets
reviewed. Check crates.io for the current version.)

| Version | Supported |
|---|---|
| Latest published `0.x` release | ✅ |
| Any older `0.x` release | ❌ — upgrade to latest |

## Reporting a vulnerability

**Please do not open a public GitHub issue for a suspected security vulnerability.**

Use GitHub's private vulnerability reporting instead:

1. Go to the [kremory repository](https://github.com/kgentic/kremory) → **Security** tab →
   **Report a vulnerability**.
2. This opens a private advisory visible only to the maintainers and GitHub — it does not
   disclose the issue publicly until a fix is ready.

If you're unable to use GitHub's private reporting flow, open a regular issue asking a
maintainer to set up an alternative private channel — do not include vulnerability details in
that issue.

Please include, where you can:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (a minimal repro against the `kremory` crate, `kremory-napi` binding, or
  `kremory-mcp` server — whichever component is affected).
- The `kremory` version (or commit SHA) you tested against.
- Whether the issue requires a malicious/untrusted LLM response, untrusted input text passed to
  `remember()`/`recall()`, or a specific feature flag to trigger.

## What's in scope

- The `kremory` crate (core bi-temporal graph engine, extraction, recall, dream consolidation,
  reversible mutations).
- The `kremory-napi` Node.js binding.
- The `kremory-mcp` Model Context Protocol server.
- Data-integrity or data-leakage issues: cross-namespace data leakage, a bi-temporal query
  returning facts outside their valid window, an `undo()`/`forget()` operation that doesn't
  actually remove/reverse what it claims to, or SQL injection into the underlying libSQL store.

## What's out of scope

- Vulnerabilities in a BYOM provider you wired in yourself (your LLM/embedder, your API keys) —
  report those to the provider.
- Vulnerabilities in upstream dependencies (`libsql`, `autoagents-llm`, etc.) — please also
  report those upstream; we'll track the dependency bump here.
- Issues that require local filesystem access to the SQLite/libSQL database file kremory already
  trusts (kremory is an embedded library — the process running it already has that access by
  design).

## Response expectations

kremory is currently maintained by a small team without a dedicated security response SLA.
We aim to acknowledge new reports within a few days and will keep you updated as we investigate.
Given the pre-1.0 status, please treat all APIs as subject to change as part of a fix.

## Disclosure

We prefer coordinated disclosure: please give us a reasonable window to ship a fix and publish
a new crate version before public disclosure. We'll credit reporters (unless you'd prefer to
stay anonymous) in the fix's changelog entry / GitHub Security Advisory.
