# RQL

**Pure Rust bi-temporal knowledge graph engine + agent memory orchestration.**

- `rql-core` — Graphiti-equivalent graph engine. LLM entity/edge extraction with temporal validity, entity dedup, contradiction handling, hybrid retrieval (semantic + BM25 + graph), community detection. **Apache-2.0**.
- `rql-memory` — Zep-equivalent orchestration over `rql-core`. Multi-tenant scoping, packaged "dream-phase" batch consolidation recipe, opinionated retrieval defaults. **Dual-license** (Apache-2.0 OSS + commercial).
- `rqlm-mcp` — MCP server wrapping `rql-memory`'s public surface. JSON-RPC tools over rmcp stdio. _(Scaffold; handler bodies land at D.4b.)_

Status: pre-v0.1.0. Workspace extracted from `the-host-application` monorepo for clean OSS development context.

## Repo layout

```
crates/
  rqlc/        rql-core (Apache-2.0)
  rqlm/        rql-memory (dual-license)
  rqlm-mcp/    MCP bridge (excluded from workspace until D.4b)
```

Directories use the typing-shorthand names (`rqlc`/`rqlm`); canonical crate names (`rql-core`/`rql-memory`) appear in `Cargo.toml` `name = ` fields, docs, and crates.io publishes. Directories will rename to canonical at v0.1.0 publish moment via `git mv` (cheap because the repo is born external).

## Development

This workspace is consumed by [the-host-application](https://github.com/kgentic/the-host-application) (private) via sibling-dir path-deps during dev:

```toml
rql-core   = { path = "../../kgentic-rql/crates/rqlc" }
rql-memory = { path = "../../kgentic-rql/crates/rqlm" }
```

Post-v0.1.0 publish, the-host-application swaps to crates.io registry deps.

## License

- `rql-core` — Apache-2.0 (`LICENSE-APACHE`)
- `rql-memory` — Apache-2.0 + commercial dual-license (`LICENSE-APACHE` + `LICENSE-COMMERCIAL`)

## Status

This README is a placeholder. v0.1.0 launch ADR sequence runs autonomously via Ralph Loop from `.ai-docs/plans/rql/plan-rqlcm-v0.1.0-ralph-loop-execution-runbook-2026-05-20.md`.
