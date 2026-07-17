# kremory

> **The SQLite of agent memory.** Embeddable, bi-temporal knowledge-graph memory for AI agents — a single Rust crate you link in, not a service you call out to.

[![crates.io](https://img.shields.io/crates/v/kremory.svg)](https://crates.io/crates/kremory)
[![docs.rs](https://img.shields.io/docsrs/kremory)](https://docs.rs/kremory)
[![license](https://img.shields.io/crates/l/kremory.svg)](LICENSE)
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

Supporting capabilities: **bi-temporal** facts (ask "what did the agent know, and when" via
`.as_of(ts)`), and **BYOM** — bring your own LLM + embedder; kremory bundles no model weights.

## Install

```toml
[dependencies]
kremory = "0.5"
```

That's it — no git dependency, no `[patch.crates-io]` stanza. kremory builds against the
published `autoagents-llm` (the model id flows as plain data, not a trait accessor). No model
weights are bundled; bring your own LLM + embedder (see [BYOM](#byom--bring-your-own-model)).

---

## Quickstart

This mirrors [`crates/kremory/examples/quickstart.rs`](crates/kremory/examples/quickstart.rs)
(same API calls, embedder body simplified for readability) — that file gets compiled by
`cargo test` (and directly via `cargo build --example quickstart`), so the API shape here is
checked against the real crate, not hand-typed and left to drift. It wires a real local Ollama
model (BYOM) and a custom embedder (BYOE):

```rust
use std::sync::Arc;
use autoagents_llm::{backends::ollama::Ollama, builder::LLMBuilder};
use kremory::{CoreResult, EmbeddingProvider, Memory, Namespace};

/// BYOE — a real consumer calls their embedding backend (OpenAI, Ollama
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
        .with_model_id("gemma4-e2b:latest") // tells kremory which structured-output strategy to use
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

`Memory::auto("./agent.db")` (Tier 1, not shown above) is a shorter path when you're OK with
env-var provider auto-detection — see [Three-tier API](#three-tier-api-progressive-disclosure)
below.

---

## What it does

kremory is an embeddable agent memory library — the SQLite of agent memory. Add it to your Rust project with one line.

Three modules ship in a single crate:

- **`kremory::facade`** — fluent `Memory` facade: the recommended public API for most consumers.
- **`kremory::core`** — bi-temporal knowledge graph engine over libSQL. Entity extraction, deduplication, hybrid retrieval, contradiction resolution.
- **`kremory::memory`** — orchestration layer. Multi-tenant namespace scoping, dream-phase batch consolidation, opinionated retrieval defaults, BYOM embedding hook.

---

## Three-tier API (progressive disclosure)

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

kremory never bundles an embedding model. The `EmbeddingProvider` trait is the only embedding interface — a single `embed` method that takes one text and returns its vector. It uses RPITIT (`impl Future`), so **no `#[async_trait]` is needed**:

```rust
use kremory::{CoreResult, EmbeddingProvider};
use std::future::Future;

struct MyEmbedder;

impl EmbeddingProvider for MyEmbedder {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Future<Output = CoreResult<Vec<f32>>> + Send + 'a {
        async move {
            // call your model here; return the embedding vector
            Ok(vec![/* … */])
        }
    }

    // Optional: report server-side token usage if your backend exposes it.
    fn last_usage_tokens(&self) -> Option<u64> { None }
}
```

> **Tip:** if clippy's `manual_async_fn` fires on the body, either capture any `Copy`
> fields into locals *before* the `async move` block (the pattern kremory's own
> embedder impls use) or write the method as a plain `async fn embed`.

The full trait (`kremory::EmbeddingProvider`):

```rust
pub trait EmbeddingProvider: Send + Sync {
    fn embed<'a>(&'a self, text: &'a str)
        -> impl Future<Output = CoreResult<Vec<f32>>> + Send + 'a;
    fn last_usage_tokens(&self) -> Option<u64> { None }
    // `into_dyn(self) -> Arc<dyn DynEmbeddingProvider>` is provided by default.
}
```

Wire it via `.with_embedder(MyEmbedder.into_dyn())` (or `Arc::new(MyEmbedder)`). **The embedding dimension is not read from the trait** — it defaults to `384`, so if your vectors are a different size, declare it: `.embedding_dim(768)`.

Wire any provider — OpenAI, Ollama, local GGUF, sentence-transformers via HTTP, anything. Your API keys, your inference costs, your data.

Why this matters: a bundled 440MB model cannot publish to crates.io (10MB compressed limit). kremory stays under 1MB published. See [ADR-002](https://github.com/kgentic/kremory/blob/main/.ai-docs/adrs/rql/adr-002-byom-distribution-moat-2026-05-22.md).

### What about GLiNER? (when you opt into NER)

kremory ships **no LLM or embedding weights**. The one exception is GLiNER: if you enable the `ner` cargo feature **and** wire the hybrid extractor via the builder (`.with_gliner()` together with `.with_llm(...)`), kremory auto-downloads the `onnx-community/gliner_large-v2.1` INT8 model (~650 MB) from HuggingFace Hub on first use, cached locally by `hf-hub` thereafter (subsequent runs are offline).

- **Default build** (no `ner` feature): no download, no GLiNER. The default LLM extractor handles entity extraction via your BYOM LLM.
- **`--features ner` + `.with_gliner()` + `.with_llm(...)`**: GLiNER candidates + one LLM typing call (the hybrid extractor) — empirically the highest-precision path (100% mock_interview / 93.8% legal_deposition).

Threshold tunable via `KREMORY_GLINER_THRESHOLD` (default `0.5`). Lower → higher recall, more noise candidates.

### Recommended local models (Ollama)

Benchmarked on **Apple Silicon M4 Max** (2026-06-24, `mock_interview` fixture). **Latency is
M4-Max-only; precision/recall are hardware-independent** (same GGUF weights → same quality
anywhere). Reproduce with
[`scripts/model-benchmark/`](https://github.com/kgentic/kremory/blob/main/scripts/model-benchmark/README.md).

| Model | think | F1 | recall | slowest call | fits 30s budget | size | role |
|---|---|---|---|---|---|---|---|
| **`gemma4:e4b`** | `false` | **84** | **90%** | ~16s | yes | 9.6GB | **`with_ollama` default — best** |
| `qwen2.5:7b` | — | 79 | 70% | ~11s | yes | 4.7GB | lighter alternative |
| `qwen2.5:14b` | — | 82 | 70% | 29–50s | no | 9.0GB | deferred / quality only |
| `gemma4:e4b` | on | 75 | 90% | 44s | no | 9.6GB | reasoning HURTS extraction |

`Memory::with_ollama` defaults to **`gemma4:e4b` with reasoning disabled** (`think:false`) — the
best extraction quality that still fits the inline 30s budget. kremory extraction is
structured-output, not reasoning: leaving thinking *on* is both slower (44s/call) and *worse*
(F1 75). Pull the two models once:

```text
ollama pull gemma4:e4b
ollama pull nomic-embed-text   # embeddings, 768-dim
```

Prefer a smaller footprint? Wire the lighter model explicitly:

```rust
let mem = Memory::with_ollama_at_model(
    "http://localhost:11434",
    Some("qwen2.5:7b".into()),   // 4.7GB, F1 ~79
    "./agent.db",
).await?;
```

Or set any model via `OLLAMA_CHAT_MODEL`, or wire your own provider through `with_llm`. The
empirical ladder's source of truth is
[`tests/llm_integration.rs`](https://github.com/kgentic/kremory/blob/main/crates/kremory/tests/llm_integration.rs).
Avoid `*-mlx` tags (Apple-Silicon-only — not portable) and `gemma4:26b` (emits junk the
validator can't reject).

For embedding, `nomic-embed-text` (Ollama, 768-dim) is the kremory-tested default.

---

## Two-Clock Temporal Model

Every fact in kremory carries two independent time dimensions:

| Column | Axis | Mutability | Meaning |
|---|---|---|---|
| `recorded_at` | Transaction time | Immutable | When the system learned this fact |
| `valid_from` | Valid time | Mutable | When the fact became true in the world |
| `valid_to` | Valid time | Mutable | When the fact stopped being true (NULL = currently valid) |

This enables: **"What did the agent know at time X if asked at time Y?"**

See [ADR-003](https://github.com/kgentic/kremory/blob/main/.ai-docs/adrs/rql/adr-003-bitemporal-audit-compliance-2026-05-22.md).

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

See [docs/api.md](https://github.com/kgentic/kremory/blob/main/docs/api.md) for the complete reference:

1. Quickstart
2. Customizing the LLM/embedder
3. Namespaces + multi-tenancy
4. Ingest (`remember`)
5. Recall — incl. full-text (BM25) content search (`.content()`, opt-in `content-search`)
6. Dream phase + consolidation
6a. Reversibility — see, trust, undo
7. Forget (GDPR)
8. Async patterns (handles + polling)
9. Event sinks
10. Advanced — substrate composition
11. Bi-temporal model
12. Migration guide (currently covers up to v0.3.2 — see [CHANGELOG](crates/kremory/CHANGELOG.md) for changes since)
13. Feature flags
14. Node / napi binding

---

## Reversible dream — see, trust, undo

`dream()` consolidates the graph by default: community detection, the supersession sweep, and
fact archival are **ON and commit**. Cross-episode entity merges are the one exception — they
default to **Shadow mode** (they compute and *report* merge decisions but fuse nothing), so you
opt into actual fusion explicitly with `.cross_episode(CrossEpisodeMode::Apply)`. Watch the split
via `summary.cross_episode_would_merge` (decisions) vs `summary.cross_episode_merged` (committed).
Committing by default is safe because **every committed mutation is reversible** (ADR-073) — you
can always SEE what changed and UNDO it.

```rust
use kremory::{Memory, Namespace};

let ns = Namespace::new("agent");

// dream() consolidates by default — all ops ON, all reversible.
let summary = mem.dream().await?;
println!("communities updated: {}", summary.communities_updated);

// SEE what dream() did to one entity (newest-first) …
let history = mem.mutation_history("alice j").in_namespace(ns.clone()).await?;
for record in &history {
    println!("{}: {}", record.mutation_id, record.summary);
}

// … or list every mutation in a namespace.
let all = mem.list_mutations().in_namespace(ns.clone()).await?;

// UNDO any of them uniformly by mutation_id — the dispatcher routes by kind.
if let Some(record) = history.first() {
    let outcome = mem.undo(record.mutation_id).execute().await?;
    println!("reversed: {outcome:?}");
}
```

### The undo surface

| Method | Reverses | Notes |
|---|---|---|
| `mem.undo(mutation_id)` | **any** logged mutation | The recommended umbrella — reads the kind and dispatches. Idempotent. |
| `mem.unmerge(mutation_id)` | an `entity_merge` | Restores the split pair + records a `merge_nogood` so the next `dream()` will **not** re-merge them. |
| `mem.undo_entity_edit(mutation_id)` | an `edit_entity` (rename/retype) | Inverse FK-rekey from the snapshot. |
| `mem.undo_delete_entity(mutation_id)` | a `delete_entity` | Un-archives facts, re-inserts edges + membership. |
| `mem.undo_delete_fact(mutation_id)` | a `delete_fact` | Restores the archived fact + un-retracts neighbours. |
| `mem.unsupersede(fact_id)` | a supersession bound | Takes a `fact_id` (not a `mutation_id`). |
| `mem.restore_archived_fact(archived_fact_id)` | a P2 archive | Takes an `archived_fact_id`. |

Every undo returns an **honest outcome** (the actual counts reversed, never a bare "ok"), is **idempotent** (a second undo is a zero-count no-op, never a double-restore), and is **fully deterministic** (replayed from an in-transaction snapshot — no LLM).

**Tracked-kind boundary:** four of the eight `MutationKind`s are logged and reversible via `undo()` (`entity_merge`, `entity_edit`, `entity_delete`, `fact_delete`); the other four are reserved (not yet produced), so `list_mutations().kind(<reserved>)` is empty by construction and `undo()` on such a row is a loud `UndoUnsupportedKind`.

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
kremory = { version = "0.5", features = ["otel"] }
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

See [docs/observability.md](https://github.com/kgentic/kremory/blob/main/docs/observability.md) for full metric catalog, label schema, cardinality discipline, and dashboard examples.

---

## Storage

libSQL (Turso-compatible). Defaults to a local embedded file — no server process required. Switch to a remote Turso URL when your application needs it — the kremory API is the same either way.

---

## Compared to alternatives

A structural summary (not a benchmark) — how kremory differs in *shape*, not just numbers:

| | **kremory** | Vector DB + RAG | Graphiti / Zep | Mem0 / Cognee |
|---|---|---|---|---|
| Data model | bi-temporal knowledge graph | embeddings over text chunks | temporal knowledge graph | vector + graph |
| **Reverse a merge / edit / delete (undo)** | ✅ built-in, deterministic | — | — | — |
| Bi-temporal (world-time *and* system-time) | ✅ | — | partial (valid-time) | — |
| Runs embedded, no server process | ✅ single libSQL file | varies | needs a graph DB / cloud | SDK → cloud or self-host |
| Language / footprint | Rust, <1 MB crate | — | Python | Python |
| Bring your own model (no weights bundled) | ✅ | n/a | ✅ | ✅ |
| Self-consolidation (a "dream" pass) | ✅ | — | partial | ✅ |

The two rows nobody else fills: **reversible graph mutations** (see [Reversible dream](#reversible-dream--see-trust-undo)) and **full bi-temporal** history (see [Two-Clock Temporal Model](#two-clock-temporal-model)). "—" means *not a headline capability of that tool*, not necessarily impossible.

---

## Status & maturity

**Pre-1.0 (`0.5.x`), used in earnest but still evolving.** Correctness coverage is strong —
the full suite runs under every feature combination (default / `ner` / `content-search` /
all-features) plus real-Ollama end-to-end journeys (`remember → dream → recall → unmerge/edit/
delete/undo`). What 1.0 still needs: an API freeze, a cross-provider model matrix, and
load/concurrency/durability testing.

**API stability:** on the pre-1.0 lane, minor releases (e.g. `0.4 → 0.5`) may contain breaking
changes — pin a minor (`kremory = "0.5"`) and read the [CHANGELOG](CHANGELOG.md) before bumping.

## Node.js / MCP — not yet published

- **Node binding (`kremory-napi`).** A napi-rs binding exists in this repo (`crates/kremory-napi`)
  mirroring the Rust `Memory` facade in camelCase, including undo/reversibility. **It is not yet
  published to npm** — build it from source if you need it today. Known limitation: wiring a
  custom (JS-callback) embedder or LLM provider can hit a native teardown assertion on abrupt
  process exit (upstream napi-rs issue; tracked as TD-005b) — this gates the npm publish.
- **MCP server (`kremory-mcp`).** A Model Context Protocol server (5 tools: remember / recall /
  dream / list_mutations / undo) exists in this repo. **It is not yet published as an installable
  package** — build it from source.

## When *not* to reach for kremory

- **You want a hosted, zero-config memory API.** kremory is an *embeddable Rust crate* you wire
  your own LLM + embedder into — not a managed service. (Bring your own model is the point; it's
  also the work.)
- **You're not in Rust.** The Node binding above works but isn't on npm yet, and there's no Python
  SDK.
- **You need proven horizontal scale / high-concurrency multi-tenant *today*.** Storage is a
  single embedded libSQL writer; large-scale concurrency is on the 1.0 roadmap, not yet load-tested.
- **You just want document RAG.** A vector DB is simpler. kremory earns its keep when you need a
  *graph* that tracks change over time and lets you undo what consolidation did.

## License

Apache-2.0. See [LICENSE](LICENSE).
