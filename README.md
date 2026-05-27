# kremory

**Embed agent memory in your app. Pure Rust. BYOM. Apache-2.0.**

Pure Rust agent memory engine. Single binary. No server process. No subscription required to ship.

```toml
[dependencies]
kremory = "0.1"
```

---

## Quickstart

```rust
use kremory::{Memory, Namespace};

// Auto-detect provider from environment:
//   OLLAMA_HOST → OPENAI_API_KEY → ANTHROPIC_API_KEY → Err
let mem = Memory::auto("./agent.db")
    .default_namespace(Namespace::new("user-jim"))
    .await?;

// Ingest — blocks until Phase 2 enrichment done
mem.remember("User prefers concise replies").await?;

// Recall — returns prompt-ready text
let context: String = mem.recall("what does user prefer?").await?;

// Close (flush WAL)
mem.close().await?;
```

No model bundled. No server process. No API key required to ship.

---

## Why kremory

- **Embeddable, not a service.** One `cargo add`. No subprocess. No container. No API key required to ship. The SQLite of agent memory.
- **BYOM — Bring Your Own Model.** kremory never bundles a 440MB embedding model. Wire your own `Arc<dyn EmbeddingProvider>` — OpenAI, Ollama, local GGUF, anything. Your costs, your keys, your data.
- **Two-clock temporal model.** Every fact carries `recorded_at` (when the system learned it — immutable) and `valid_from`/`valid_to` (when it was true in the world — mutable). Audit-grade history with no data loss. Contradiction resolver lands in v0.1.1.

---

## Three-tier API

```
Tier 1 — Just works      Memory::auto("./agent.db").await?
          ↓
Tier 2 — Customizable    Memory::open(...).with_llm(...).with_embedder(...).await?
          ↓
Tier 3 — Substrate       kremory::memory::submit_episode(...)  (advanced)
```

See [docs/api.md](docs/api.md) for the full reference.

---

## Roadmap

| Version | Milestone |
|---|---|
| **v0.1.0** | Substrate + bi-temporal storage — two-clock columns, `as_of` queries, `published_at` precedence, BYOM via `Arc<dyn ChatProvider>` + `EmbeddingProvider`, libSQL, `Memory` facade, single-process |
| **v0.1.1** | Moat + Phase trait — contradiction engine, multi-process dream lock, `otel` feature complete, Phase trait + DreamCycle replaces PASSES, distillation primitives |
| **v0.1.2** | Show HN release — comparison matrix, CI badges, 3 runnable examples |
| **v0.2.0** | Hybrid recall + `kremory-mcp` — basic RRF + tunable `SearchOpts`, MCP server crate first publishable release |
| **v0.2+** | `kremory-cli` — command-line interface |
| **v0.3+** | `kremory-claude-plugin` — native Claude Code plugin |
| **v0.4+** | `codebase-memgraph-kremory` — fork of codebase-memory-mcp replacing C storage with kremory |

---

## Compared to alternatives

See [docs/comparison.md](docs/comparison.md) for a full matrix: kremory vs codemem vs codebase-memory-mcp vs Mem0 vs Letta vs Zep/Graphiti.

---

## Architecture

```
kremory (OSS, Apache-2.0) — single crate, crates.io
  kremory::facade  — Memory facade (Tier 1 / Tier 2 consumer surface)
  kremory::core    — bi-temporal graph engine, libSQL substrate
  kremory::memory  — orchestration: multi-tenant scoping, dream-phase consolidation

Reference consumers (OSS, kgentic-owned, deferred versions):
  kremory-mcp    — MCP server wrapping kremory::memory (v0.2.0)
  kremory-cli    — command-line interface (v0.2+)
  kremory-claude-plugin — native Claude Code plugin (v0.3+)
  codebase-memgraph-kremory — fork replacing C storage layer (v0.4+)
```

---

## Repo layout

```
crates/
  kremory/       single crate: kremory::facade + kremory::core + kremory::memory (Apache-2.0)
    src/
      facade/    Memory facade — Tier 1 / Tier 2 consumer surface
      core/      bi-temporal graph primitives
      memory/    orchestration layer
  kremory-mcp/   MCP bridge (excluded from workspace until v0.2.0)
docs/
  api.md         Full API reference (facade-first)
  comparison.md  kremory vs alternatives
.ai-docs/
  adrs/          Architecture Decision Records
  architecture/  Architecture spec
  plans/         Execution runbooks
  research/      Competitive research + market analysis
```

---

## License

`kremory` — Apache-2.0. See [crates/kremory/LICENSE](crates/kremory/LICENSE).

---

## Research foundations

kremory's bi-temporal storage model and contradiction-resolution architecture draw from:

- Helms et al., "Zep: A Temporal Knowledge Graph Architecture for Agent Memory" (2025) — arXiv:2501.13956

---

## Status

Pre-v0.1.0. Launch sequence runs from `.ai-docs/plans/rql/plan-rqlcm-v0.1.0-ralph-loop-execution-runbook-2026-05-20.md`.
