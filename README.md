# kremory

**Pure Rust bi-temporal knowledge graph engine + agent memory orchestration.**

- `kremory::core` — Graphiti-equivalent graph engine. LLM entity/edge extraction with temporal validity, entity dedup, contradiction handling, hybrid retrieval (semantic + BM25 + graph), community detection. **Apache-2.0**.
- `kremory::memory` — Zep-equivalent orchestration over `kremory::core`. Multi-tenant scoping, packaged "dream-phase" batch consolidation recipe, opinionated retrieval defaults. **Apache-2.0**.
- `kremory-mcp` — MCP server wrapping `kremory::memory`'s public surface. JSON-RPC tools over rmcp stdio. _(Scaffold; handler bodies land at D.4b.)_

Status: pre-v0.1.0. Workspace extracted from `the-host-application` monorepo for clean OSS development context.

## Repo layout

```
crates/
  kremory/       single merged crate: kremory::core + kremory::memory (Apache-2.0)
    src/
      core/      bi-temporal graph primitives (was rql-core)
      memory/    orchestration layer (was rql-memory)
  kremory-mcp/   MCP bridge (excluded from workspace until D.4b)
```

## Development

This workspace is consumed by [the-host-application](https://github.com/kgentic/the-host-application) (private) via sibling-dir path-deps during dev:

```toml
kremory = { path = "../../kgentic-kremory/crates/kremory" }
```

Post-v0.1.0 publish, the-host-application swaps to crates.io registry deps.

## License

`kremory` — Apache-2.0

## Status

This README is a placeholder. v0.1.0 launch ADR sequence runs autonomously via Ralph Loop from `.ai-docs/plans/rql/plan-rqlcm-v0.1.0-ralph-loop-execution-runbook-2026-05-20.md`.
