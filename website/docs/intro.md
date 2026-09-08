---
sidebar_position: 1
slug: /
---

# kremory

> **The SQLite of agent memory.** Embeddable, bi-temporal knowledge-graph memory for AI agents — a single Rust crate you link in, not a service you call out to.

[![crates.io](https://img.shields.io/crates/v/kremory.svg)](https://crates.io/crates/kremory)
[![docs.rs](https://img.shields.io/docsrs/kremory)](https://docs.rs/kremory)
[![license](https://img.shields.io/crates/l/kremory.svg)](https://github.com/kgentic/kremory/blob/main/LICENSE)
![MSRV](https://img.shields.io/badge/MSRV-1.86-blue)

kremory ships as a library, not a service: one crate, one embedded libSQL file, no server process,
no subscription to ship. Two things it does that (as far as we've checked) no other agent-memory
tool does:

- **Local-first.** Runs entirely on your machine — no server, no API key, no data leaving the box.
  Point the same API at a remote Turso URL later if you need to; nothing changes but the connection
  string.
- **Memory you can undo.** Every merge, edit, and delete kremory's background consolidation
  ("dream") phase makes is logged and reversible — `mem.undo(mutation_id)` reverses it, deterministically,
  no LLM involved. Nothing kremory writes is a silent overwrite.

Supporting capabilities: **bi-temporal** facts (ask "what was true at time t" via
`.as_of(ts)` — and keep the second clock, `recorded_at`, on every row for audit), and
**BYOM** — bring your own LLM + embedder; kremory bundles no model weights.

```toml
[dependencies]
kremory = "0.7"
```

No model weights bundled. No server process. No API key required to ship.

## Get started

The fastest way to see kremory working is the **[Getting Started guide](./getting-started.md)** —
a real, captured transcript of `cargo add kremory` against the published crate through a working
`remember` → `recall`, in a few minutes, with no checkout of this repo required.

## Learn more

- **[Getting Started](./getting-started.md)** — install the published crate and run a real program.
- **[API Reference](./api/index.md)** — the complete surface: builder customization,
  namespaces, dream/consolidation, undo, feature flags, the Node binding.
- **[Comparison](./comparison.md)** — how kremory differs from Graphiti/Zep, Mem0, codemem, and
  friends.
- **[Benchmarks](./benchmarks.md)** — the LoCoMo results and methodology.

## Status

**Pre-1.0 (`0.7.x`), used in earnest but still evolving.** On the pre-1.0 lane, minor releases may
contain breaking changes — pin a minor (`kremory = "0.7"`) and read the
[CHANGELOG](https://github.com/kgentic/kremory/blob/main/CHANGELOG.md) before bumping. See the
[Getting Started guide](./getting-started.md) for what's ready today, including the current state
of the Node binding and MCP server (neither is published yet — build from source).

## License

Apache-2.0. See [LICENSE](https://github.com/kgentic/kremory/blob/main/LICENSE).
