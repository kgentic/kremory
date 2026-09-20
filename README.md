# kremory

> **The SQLite of agent memory.** Embeddable, bi-temporal knowledge-graph memory for AI agents — a single Rust crate you link in, not a service you call out to.

[![crates.io](https://img.shields.io/crates/v/kremory.svg)](https://crates.io/crates/kremory)
[![docs.rs](https://img.shields.io/docsrs/kremory)](https://docs.rs/kremory)
[![docs](https://img.shields.io/badge/docs-kgentic.github.io%2Fkremory-blue)](https://kgentic.github.io/kremory/)
[![license](https://img.shields.io/crates/l/kremory.svg)](LICENSE)
![MSRV](https://img.shields.io/badge/MSRV-1.86-blue)

kremory ships as a library, not a service: one crate, one embedded libSQL file, no server process,
no subscription to ship. Two things it does that (as far as we've checked) no other agent-memory
tool does:

- **Local-first.** Runs entirely on your machine — no server, no API key, no data leaving the box.
  Point the same API at a remote Turso URL later if you need to; nothing changes but the connection
  string.
- **Memory you can undo.** Every merge, edit, and delete kremory's background consolidation
  ("dream") phase makes is logged and reversible — `mem.undo(mutation_id)` reverses it,
  deterministically, no LLM involved. Nothing kremory writes is a silent overwrite.

Supporting capabilities: **bi-temporal** facts (ask "what was true at time t" via `.as_of(ts)` —
and keep the second clock, `recorded_at`, on every row for audit), and **BYOM** — bring your own
LLM + embedder; kremory bundles no model weights.

## Install

```toml
[dependencies]
kremory = "0.8"

# kremory's API is async and returns errors, so a runtime and an error type are
# needed to run the Quickstart below.
tokio = { version = "1", features = ["full"] }
anyhow = "1"

# The Quickstart below builds an Ollama provider directly (BYOM). kremory depends
# on autoagents-llm internally but does NOT re-export it, so constructing a
# provider yourself needs it as a direct dependency, with the backend feature you
# use. Not needed if you stick to `Memory::auto` / `Memory::with_ollama`, which
# build the provider for you.
autoagents-llm = { version = "0.3", features = ["ollama"] }
```

No git dependency and no `[patch.crates-io]` stanza — kremory builds against the published
`autoagents-llm`. No model weights are bundled.

## Quickstart

**Want to see it work with zero setup first?** `cargo run --example offline_remember_recall` needs
no Ollama, no API key and no network — it is one of 21 offline examples that are executed (not just
compiled) by this repo's own guard, so it either runs or the build is red.

The example below is the real thing: a local Ollama model (BYOM) and a custom embedder. It mirrors
[`crates/kremory/examples/quickstart.rs`](crates/kremory/examples/quickstart.rs), which is compiled
by `cargo test`.

```rust
use std::sync::Arc;
use autoagents_llm::{backends::ollama::Ollama, builder::LLMBuilder};
use kremory::{CoreResult, EmbeddingProvider, Memory, Namespace};

/// A real consumer calls their embedding backend (OpenAI, Ollama
/// `nomic-embed-text`, a local GGUF, …) inside `embed`. This one is a
/// deterministic hash so the example needs no network embedding model.
struct DemoEmbedder;

impl EmbeddingProvider for DemoEmbedder {
    fn embed<'a>(&'a self, text: &'a str)
        -> impl std::future::Future<Output = CoreResult<Vec<f32>>> + Send + 'a
    {
        async move { Ok(vec![text.len() as f32; 16]) } // 16-dim demo vector
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // BYOM — bring your own chat provider. Real Ollama here, built via autoagents-llm.
    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url("http://localhost:11434")
        .model("gemma4-e2b:latest")
        .timeout_seconds(120)
        .build()?;

    // Tier 2 builder — type-state guarded: `.await` won't compile until both
    // `.with_llm()` and `.with_embedder()` are set.
    let mem = Memory::open("./agent.db")
        .embedding_dim(16) // our embedder is 16-dim, not the 384 default
        .default_namespace(Namespace::new("quickstart"))
        .with_llm(llm)
        .with_model_id("gemma4-e2b:latest") // which structured-output strategy to use
        .with_embedder(DemoEmbedder.into_dyn())
        .await?;

    // Ingest — runs real LLM extraction.
    mem.remember("Jim prefers concise replies and writes Rust.").await?;

    // Recall — returns prompt-ready text.
    let ctx = mem.recall("what language does Jim use?").await?;
    println!("{ctx}");

    mem.close().await?; // flush WAL
    Ok(())
}
```

No model bundled. No server process. No API key required to ship.

> ⚠️ **Ingest is LLM-bound and takes seconds, not milliseconds** — measured at ~37.5 s/session
> against a cloud model, ~119 s locally. If you are putting kremory behind an HTTP handler, take
> the write off the request path with `.no_wait()`. See
> [deployment](https://kgentic.github.io/kremory/docs/deployment).

## Documentation

Full guides, API reference and 20+ runnable examples: **<https://kgentic.github.io/kremory/>**

| | |
|---|---|
| [Getting started](https://kgentic.github.io/kremory/docs/getting-started) | first 30 seconds, offline |
| [Recall](https://kgentic.github.io/kremory/docs/api/recall) · [Ingest](https://kgentic.github.io/kremory/docs/api/ingest) | the two calls you will actually use |
| [Bi-temporal](https://kgentic.github.io/kremory/docs/api/bi-temporal) | `.as_of()` and the two clocks |
| [Reversibility](https://kgentic.github.io/kremory/docs/api/reversibility) | see, trust and undo what dream did |
| [Dream](https://kgentic.github.io/kremory/docs/api/dream) | the consolidation pass |
| [Namespaces](https://kgentic.github.io/kremory/docs/api/namespaces) | multi-tenancy |
| [Feature flags](https://kgentic.github.io/kremory/docs/api/feature-flags) | what each cargo feature turns on |
| [Deployment](https://kgentic.github.io/kremory/docs/deployment) | connections, locking, async writes |
| [Observability](https://kgentic.github.io/kremory/docs/observability) | metrics + tracing |

API reference is also on [docs.rs](https://docs.rs/kremory).

## Compared to alternatives

A structural summary (not a benchmark) — how kremory differs in *shape*, not just numbers:

| | **kremory** | Vector DB + RAG | Graphiti / Zep | Mem0 / Cognee |
|---|---|---|---|---|
| Data model | bi-temporal knowledge graph | embeddings over text chunks | temporal knowledge graph | vector + graph |
| **Reverse a merge / edit / delete (undo)** | ✅ built-in, deterministic | — | — | — |
| Bi-temporal (world-time *and* system-time) | ✅ | — | partial (valid-time) | — |
| Runs embedded, no server process | ✅ single libSQL file | varies | needs a graph DB / cloud | SDK → cloud or self-host |
| Language / footprint | Rust, ~1 MB crate | — | Python | Python |
| Bring your own model (no weights bundled) | ✅ | n/a | ✅ | ✅ |
| Self-consolidation (a "dream" pass) | ✅ | — | partial | ✅ |

"—" means *not a headline capability of that tool*, not necessarily impossible.

### LoCoMo benchmark — and the caveat that makes it honest

Measured 2026-09-04 on the full 1,540-question corpus (not a sample): **kremory 92.1%** against
**Mem0 Platform 92.5%** (hosted, paid), reproducing Mem0's own protocol — `gpt-5` answerer + judge,
top-200 memories.

**This is not what you get with zero configuration.** `.recall()` defaults to `k=10`; that number
used `k=200`. Out-of-the-box is closer to **89%** (measured at `k=20`). The known weak spot is
**multi-hop questions at 74%**, against 90-95% elsewhere.

Full methodology, per-category breakdown and reproduction steps:
[`docs/benchmarks.md`](docs/benchmarks.md).

## Status & maturity

**Pre-1.0 (`0.8.x`), used in earnest but still evolving.** Correctness coverage is strong — the
full suite runs under every feature combination (default / `ner` / `content-search` / all-features)
plus real-Ollama end-to-end journeys (`remember → dream → recall → unmerge/edit/delete/undo`).
Crash-resume and transaction-rollback paths carry regression tests that have each been observed to
FAIL against the un-fixed code, not merely to pass. What 1.0 still needs: an API freeze, a
cross-provider model matrix, and **load/concurrency** testing.

**API stability:** on the pre-1.0 lane, minor releases (e.g. `0.7 → 0.8`) may contain breaking
changes — pin a minor (`kremory = "0.8"`) and read the [CHANGELOG](CHANGELOG.md) before bumping.

## Node.js / MCP — not yet published

- **Node binding (`kremory-napi`).** A napi-rs binding exists in this repo mirroring the Rust
  `Memory` facade in camelCase, including undo/reversibility. **It is not yet published to npm** —
  build it from source if you need it today. Known limitation: wiring a custom (JS-callback)
  embedder or LLM provider can hit a native teardown assertion on abrupt process exit (upstream
  napi-rs issue) — this gates the npm publish.
- **MCP server (`kremory-mcp`).** A Model Context Protocol server (5 tools: remember / recall /
  dream / list_mutations / undo) exists in this repo. **It is not yet published as an installable
  package** — build it from source.

## When *not* to reach for kremory

- **You want a hosted, zero-config memory API.** kremory is an *embeddable Rust crate* you wire
  your own LLM + embedder into — not a managed service.
- **You're not in Rust.** The Node binding above works but isn't on npm yet, and there's no Python
  SDK.
- **You need proven horizontal scale / high-concurrency multi-tenant *today*.** Storage is a single
  embedded libSQL writer; large-scale concurrency is on the 1.0 roadmap, not yet load-tested.
- **You just want document RAG.** A vector DB is simpler. kremory earns its keep when you need a
  *graph* that tracks change over time and lets you undo what consolidation did.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security policy: [SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
