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

// Ingest — blocks until Phase 2 enrichment done (~500ms typical)
mem.remember("User prefers concise replies").await?;

// Recall — returns prompt-ready text
let context: String = mem.recall("what does user prefer?").await?;

// Dream (consolidation) — blocks until done
let summary = mem.dream().await?;
println!("communities updated: {}", summary.communities_updated);

// Forget (GDPR-style delete)
let deleted = mem.forget().execute().await?;

// Close (flush WAL)
mem.close().await?;
```

No model bundled. No server process. No API key required to ship.

---

## What it does

kremory is an embeddable agent memory library — the SQLite of agent memory. Add it to your Rust project with one line.

Three modules ship in a single crate:

- **`kremory::facade`** — fluent `Memory` facade: the recommended public API for most consumers.
- **`kremory::core`** — bi-temporal knowledge graph engine over libSQL. Entity extraction, deduplication, hybrid retrieval, contradiction resolution.
- **`kremory::memory`** — orchestration layer. Multi-tenant namespace scoping, dream-phase batch consolidation, opinionated retrieval defaults, BYOM embedding hook.

---

## Three-tier API (React philosophy)

```
Tier 1 — Just works

    let mem = Memory::auto("./agent.db").await?;

Tier 1.5 — Named shortcuts (auto-wrap providers with observability)

    let mem = Memory::with_ollama("./agent.db").await?;
    let mem = Memory::with_openai("./agent.db").await?;   // OPENAI_API_KEY required
    let mem = Memory::with_anthropic("./agent.db").await?;

Tier 2 — Customizable builder

    let mem = Memory::open("./agent.db")
        .with_llm_tracked("openai", "gpt-4o-mini", Arc::new(my_llm))  // wraps with metric emission
        .with_embedder(Arc::new(my_embedder))
        .default_namespace(Namespace::new("acme-corp"))
        .await?;

    // Or without observability:
    let mem = Memory::open("./agent.db")
        .with_llm(Arc::new(my_llm))
        .with_embedder(Arc::new(my_embedder))
        .await?;

Tier 3 — Substrate composition (advanced)

    use kremory::memory::{submit_episode, search, context_block};
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

This enables: **"What did the agent know at time X if asked at time Y?"**

See [ADR-003](../../.ai-docs/adrs/rql/adr-003-bitemporal-audit-compliance-2026-05-22.md).

---

## Namespaces + multi-tenancy

```rust
// Single tenant
let mem = Memory::auto("./agent.db")
    .default_namespace(Namespace::new("my-agent"))
    .await?;
mem.remember("User prefers dark mode").await?;  // uses default namespace

// Multi-tenant: per-call namespace override
mem.remember("Tenant A data")
    .in_namespace(Namespace::new("tenant-a"))
    .await?;

mem.remember("Q4 support ticket")
    .in_namespace(Namespace::new("acme-corp").with_thread("support-q4"))
    .await?;

// Each namespace is fully isolated — no cross-tenant leakage
let ctx = mem.recall("user preferences")
    .in_namespace(Namespace::new("tenant-a"))
    .await?;
```

---

## Full API reference

See [docs/api.md](../../docs/api.md) for the complete reference covering all 12 sections:

1. Quickstart
2. Customizing the LLM/embedder
3. Namespaces + multi-tenancy
4. Ingest (`remember`)
5. Recall
6. Dream phase + consolidation
7. Forget (GDPR)
8. Async patterns (handles + polling)
9. Event sinks
10. Advanced — substrate composition
11. Bi-temporal model
12. Migration guide

---

## Observability (v0.1.2+)

kremory emits structured metrics + tracing spans for every LLM and embedding call when you wire BYOM providers via Tier 1 shortcuts or `with_llm_tracked`. The Tier 1 shortcuts (`with_ollama`, `with_openai`, `with_anthropic`) auto-wrap providers in `TokenTrackingChatProvider` internally.

**Emitted metrics** (via the [`metrics`](https://crates.io/crates/metrics) crate):

| Metric | Type | Labels | Notes |
|---|---|---|---|
| `kremory_core_tokens_total` | Counter | `operation` (`chat`/`embed`), `provider`, `model`, `direction` (`input`/`output`) | Cumulative token counts |
| `kremory_core_cost_usd_total` | Gauge (f64 USD) | `operation`, `provider`, `model` | Cumulative cost in USD. Computed from `kremory::core::rates::PROVIDER_RATES` lookup |
| `kremory_core_chat_duration_seconds` | Histogram | `provider`, `model`, `status` (`ok`/`error`) | Per-call wall-clock duration |

**`error.type` is a tracing span attribute** (not a histogram label) bounded to `{"server_error", "client_error", "parse_error"}` per ADR D7 cardinality discipline. Query via Langfuse / Phoenix / OTLP span explorers.

**Optional OTLP export** — enable the `otel` cargo feature:

```toml
kremory = { version = "0.1", features = ["otel"] }
```

```rust
use kremory::observability::{init_telemetry, TelemetryConfig};

let handle = init_telemetry(TelemetryConfig::default())?;
// `OTEL_EXPORTER_OTLP_ENDPOINT` env var (default http://localhost:4317) configures the OTLP endpoint
// `handle` keeps the OTel TracerProvider alive — drop it at shutdown via `handle.shutdown()` to flush spans
```

**Override bundled rates** — supply your own `provider-rates.toml`:

```rust
let mem = Memory::open("./agent.db")
    .with_provider_rates_path("./my-rates.toml")
    .with_llm_tracked("openai", "gpt-4o-mini", Arc::new(my_llm))
    .with_embedder(Arc::new(my_embedder))
    .await?;
```

See [docs/observability.md](../../docs/observability.md) for full metric catalog, label schema, cardinality discipline, and dashboard examples.

---

## Storage

libSQL (Turso-compatible). Defaults to a local embedded file — no server process required. Switch to a remote Turso URL when your application needs it — the kremory API is the same either way.

---

## Compared to alternatives

See [docs/comparison.md](../../docs/comparison.md) for the full matrix.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
