# kremory

**Embed agent memory in your app. Pure Rust. BYOM. Apache-2.0.**

Pure Rust agent memory engine. Single binary. No server process. No subscription required to ship.

```toml
[dependencies]
kremory = "0.1"
```

---

## What it does

kremory is an embeddable agent memory library — the SQLite of agent memory. Add it to your Rust project with one line. No server process. No container. No subscription required to ship.

Two modules ship in a single crate:

- **`kremory::core`** — bi-temporal knowledge graph engine over libSQL. Entity extraction, deduplication, hybrid retrieval (semantic + BM25 + graph), contradiction resolution with two-clock temporal model.
- **`kremory::memory`** — orchestration layer. Multi-tenant workspace scoping, dream-phase batch consolidation, opinionated retrieval defaults, BYOM embedding hook.

---

## Quickstart

```rust
use kremory::memory::{MemoryHandle, WorkspaceScope, SubmitOpts, EpisodeRef};
use std::sync::Arc;

// Bring your own embedder — kremory never bundles a model
let embedder: Arc<dyn kremory::core::EmbeddingProvider> = your_embedder();

// Open a local database (libSQL file, no server required)
let handle = MemoryHandle::open("./agent.db", embedder).await?;
let scope   = WorkspaceScope::new("my-agent");

// Ingest a fact
handle.submit_episode(
    EpisodeRef::new("Alice decided the team will use async channels over shared state"),
    SubmitOpts::default(),
).await?;

// Retrieve relevant context for an LLM prompt
let ctx = handle.context_block(&scope, "what architecture decisions were made?").await?;
println!("{ctx}");  // ready to inject into your prompt
```

---

## BYOM — Bring Your Own Model

kremory never bundles an embedding model. The `EmbeddingProvider` trait is the only embedding interface:

```rust
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;
    fn dimensions(&self) -> usize;
    fn last_usage_tokens(&self) -> Option<u64> { None }
}
```

Wire any provider — OpenAI, Ollama, local GGUF, sentence-transformers via HTTP, anything. Your API keys, your inference costs, your data.

Why this matters: a bundled 440MB model cannot publish to crates.io (10MB compressed limit). kremory stays under 1MB published. See [ADR-002](../../.ai-docs/adrs/rql/adr-002-byom-distribution-moat-2026-05-22.md).

---

## Two-Clock Temporal Model

Every fact in kremory carries two independent time dimensions:

| Column | Axis | Mutability | Meaning |
|---|---|---|---|
| `recorded_at` | Transaction time | Immutable | When the system learned this fact |
| `valid_from` | Valid time | Mutable | When the fact became true in the world |
| `valid_to` | Valid time | Mutable | When the fact stopped being true (NULL = currently valid) |

This enables the canonical audit query: **"What did the agent know at time X if asked at time Y?"**

```sql
SELECT * FROM entities
WHERE recorded_at <= :tx_time_Y
  AND valid_from  <= :valid_time_X
  AND (valid_to IS NULL OR valid_to > :valid_time_X)
```

Active contradiction resolver: when a new episode conflicts with an existing fact, the resolver sets `valid_to` on the old record (does not delete it) and records the resolution strategy: `Superseded`, `Merged`, `Forked`, or `Ignored`.

See [ADR-003](../../.ai-docs/adrs/rql/adr-003-bitemporal-audit-compliance-2026-05-22.md).

---

## Storage

libSQL (Turso-compatible). Defaults to a local embedded file — no server process required. Switch to a remote Turso URL when your application needs it — the kremory API is the same either way.

---

## What ships at v0.1.0

- `kremory::core` — bi-temporal graph engine, libSQL storage, entity + edge schema, hybrid retrieval, contradiction resolver
- `kremory::memory` — `MemoryHandle`, `WorkspaceScope`, `submit_episode`, `context_block`, dream-phase API, BYOM embedding hook
- `EmbeddingProvider` trait — wire any provider

Not in v0.1.0 (coming in v0.2.0+): Cypher query language, MCP server crate (`kremory-mcp`), CLI (`kremory-cli`), native IDE plugin.

---

## Compared to alternatives

See [docs/comparison.md](../../docs/comparison.md) for the full matrix.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
