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
- **BYOM — Bring Your Own Model.** kremory never bundles a 440MB embedding model. Wire your own `Arc<dyn EmbeddingProvider>` and `Arc<dyn ChatProvider>` — OpenAI, Anthropic, Ollama, local GGUF, anything. Your costs, your keys, your data.
- **Two-clock temporal model.** Every fact carries `recorded_at` (when the system learned it — immutable) and `valid_from`/`valid_to` (when it was true in the world — mutable). Audit-grade history with no data loss.
- **First-class observability.** Token counters, cumulative USD cost gauges, duration histograms, OpenTelemetry GenAI SemConv spans — emitted automatically when you wire BYOM providers via Tier 1 shortcuts or `with_llm_tracked` builder. Optional OTLP exporter behind the `otel` cargo feature.

---

## Three-tier API

```
Tier 1 — Just works      Memory::auto("./agent.db").await?
          ↓
Tier 2 — Customizable    Memory::open(...).with_llm_tracked(...).with_embedder(...).await?
          ↓
Tier 3 — Substrate       kremory::memory::submit_episode(...)  (advanced)
```

See [docs/api.md](docs/api.md) for the full reference and [docs/observability.md](docs/observability.md) for the metrics + spans surface.

---

## Roadmap

| Version | Status | Milestone |
|---|---|---|
| **v0.1.0** | ✓ shipped 2026-05-27 | Substrate + bi-temporal storage — two-clock columns, `as_of` queries, `published_at` precedence, BYOM via `Arc<dyn ChatProvider>` + `Arc<dyn EmbeddingProvider>`, libSQL, `Memory` facade, single-process |
| **v0.1.1** | ✓ shipped 2026-05-27 | Recall pipeline redesign — 5 substrate bugs fixed (RRF k=60, episodic_edges authoritative, LightRAG stub-entity, first-mention snippet, `SourceKind::Episode`) |
| **v0.1.2** | ✓ shipped 2026-05-28 | **LLM observability parity** — `TokenTrackingChatProvider`, `ProviderRates` loader + cost emission, `tracing::instrument` on Engine, bounded `error.type` span attribute, `with_llm_tracked` builder, real OTel exporter behind `otel` feature |
| **v0.1.3** | ✓ shipped 2026-05-28 | Hygiene — `cargo clippy --all-targets --all-features` clean |
| **v0.1.4** | planned | Anthropic prompt-cache token tracking + examples directory + doc-vs-code parity + AA upstream PRs (Ollama + Google usage) |
| **v0.1.5** | planned | Quality eval harness + criterion benchmarks |
| **v0.2.0** | planned | Cargo features (`substrate` / `facade` / `ingest` / `tier1-providers`) per [ADR-028](.ai-docs/adrs/rql/adr-028-defer-crate-split-cargo-features-2026-05-28.md) — single-crate stays; consumer-asymmetry via features, not crate split |
| **v0.3.0** | planned | `kremory::ingest` module (NOT crate per ADR-028) — `DocumentSource` / `Chunker` / `Enricher` traits + minimal default impls |
| **v0.4+** | tentative | `kremory-mcp` first publishable release; reference consumers (`kremory-cli`, claude-plugin, codebase-memgraph-kremory) |

Full follow-up manifest: [.ai-docs/planning/roadmap-post-v013-2026-05-28.md](.ai-docs/planning/roadmap-post-v013-2026-05-28.md).

---

## Compared to alternatives

See [docs/comparison.md](docs/comparison.md) for a full matrix: kremory vs codemem vs codebase-memory-mcp vs Mem0 vs Letta vs Zep/Graphiti.

---

## Architecture

```
kremory (OSS, Apache-2.0) — single crate, crates.io
  kremory::facade        — Memory facade (Tier 1 / Tier 2 consumer surface)
  kremory::core          — bi-temporal graph engine, libSQL substrate, chat_tracking + rates
  kremory::memory        — orchestration: multi-tenant scoping, dream-phase consolidation
  kremory::observability — re-exports for TokenTrackingChatProvider, ProviderRates, init_telemetry

Reference consumers (OSS, kgentic-owned, deferred versions):
  kremory-mcp    — MCP server wrapping kremory::memory (in tree, deferred publish)
  kremory-cli    — command-line interface (post v0.2.0)
  codebase-memgraph-kremory — fork replacing C storage layer (post v0.2.0)
```

Per [ADR-028](.ai-docs/adrs/rql/adr-028-defer-crate-split-cargo-features-2026-05-28.md), kremory stays single-crate. Consumer asymmetry (substrate-only consumers vs full-organism consumers) will be expressed via cargo features at v0.2.0, not a multi-crate split.

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
  kremory-eval/  internal quality evaluation harness (publish = false) — see docs/eval.md
docs/
  api.md            Full API reference (facade-first)
  comparison.md     kremory vs alternatives
  eval.md           Quality evaluation harness (LongMemEval + RAGAS + graph integrity)
  eval-fixtures.md  Eval fixture inventory
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

**Currently shipping**: v0.1.3 (live on crates.io, published 2026-05-28). Early-adopter ready — Quickstart code path verified end-to-end. Self-serve readiness still in progress (examples directory + observability narrative doc land in v0.1.4).

**Stability promise**: pre-1.0 minor versions may introduce additive changes; breaking changes will bump to v0.2.0 with a published migration guide. BYOM contract (`Arc<dyn ChatProvider>` + `Arc<dyn EmbeddingProvider>`) and the `Memory::auto`/`Memory::open` Tier-1/Tier-2 surface are stability commitments through v0.1.x.

Detailed release plan + open items: [.ai-docs/planning/roadmap-post-v013-2026-05-28.md](.ai-docs/planning/roadmap-post-v013-2026-05-28.md).
