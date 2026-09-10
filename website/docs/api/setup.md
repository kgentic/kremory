# Setup — quickstart and providers

## Quickstart

The fastest path to working agent memory. No provider configuration required when environment
variables are set.

```rust
use kremory::{Memory, Namespace};

// Auto-detect provider from environment:
//   $OLLAMA_HOST        → Ollama (gemma4:e4b, reasoning disabled + nomic-embed-text)
//   $OPENAI_API_KEY     → OpenAI (gpt-4o-mini + text-embedding-3-small)
//   $ANTHROPIC_API_KEY  → Anthropic LLM + deterministic embedder fallback (warns)
//   (none)              → Err(Error::NoProviderConfigured) — helpful message included
let mem = Memory::auto("./agent.db").await?;
let ns = Namespace::new("user-jim");

// Ingest — blocks until Phase 2 enrichment done (~500ms typical)
mem.remember("User prefers concise replies").in_namespace(ns.clone()).await?;

// Recall — returns prompt-ready context string
let context: String = mem.recall("what does user prefer?").in_namespace(ns.clone()).await?;

// Dream (consolidation) — blocks until done (~5–60s depending on corpus)
let summary = mem.dream().in_namespace(ns.clone()).execute().await?;
println!("communities updated: {}", summary.communities_updated);

// Forget (GDPR-style delete of everything in this namespace — graph AND source text)
let erased = mem.forget().in_namespace(ns.clone()).execute().await?;
println!("erased {} episodes, {} facts", erased.episodes, erased.facts);

// Explicit close (flushes WAL)
mem.close().await?;
```

`Memory` clones cheaply — it wraps an `Arc` internally:

```rust
let mem2 = mem.clone();   // cheap — Arc clone
tokio::spawn(async move { mem2.remember("Background task").await });
```

---

## Customizing the LLM/embedder

### Tier 1.5 — Named shortcuts

Skip environment detection; use a named provider directly.

```rust
// Ollama at localhost:11434 (default models: gemma4:e4b with reasoning disabled + nomic-embed-text)
let mem = Memory::with_ollama("./agent.db").await?;

// Ollama at a custom URL (useful for remote GPU machines)
let mem = Memory::with_ollama_at("http://192.168.1.5:11434", "./agent.db").await?;

// OpenAI — requires $OPENAI_API_KEY; uses gpt-4o-mini + text-embedding-3-small
let mem = Memory::with_openai("./agent.db").await?;

// Anthropic — requires $ANTHROPIC_API_KEY. LLM = claude-3-haiku.
// ⚠️ NO EMBEDDING MODEL — see the warning below this block.
let mem = Memory::with_anthropic("./agent.db").await?;
```

### Tier 2 — Builder (full control)

```rust
use kremory::{Memory, Namespace, DynEmbeddingProvider};
use std::sync::Arc;

let mem = Memory::open("./agent.db")
    .with_llm(Arc::new(my_llm))           // Arc<dyn ChatProvider> — required (untracked)
    .with_embedder(Arc::new(my_embedder)) // Arc<dyn DynEmbeddingProvider> — required
    .with_event_sink(Arc::new(MySink))    // Arc<dyn EnrichmentEventSink> — optional
    .default_namespace(Namespace::new("acme-corp"))  // optional
    .await?;
```

The builder is type-state guarded: `.await` on a `MemoryBuilder` without calling both
`.with_llm()` (optionally followed by `.with_token_tracking()`) and `.with_embedder()` is a **compile error**, not a runtime error.

```rust,compile_fail
// Compile error — missing .with_embedder()
let mem = Memory::open("./agent.db").with_llm(llm).await?; // ERROR
```

### Tier 2 — Builder with observability (v0.1.2+)

For automatic token + cost + duration metric emission, add `.with_token_tracking(provider, model)` after `.with_llm(...)`:

```rust
let mem = Memory::open("./agent.db")
    .with_llm(Arc::new(my_llm))
    .with_token_tracking("openai", "gpt-4o-mini")  // explicit labels
    .with_embedder(Arc::new(my_embedder))
    .await?;
```

### ⚠️ `with_anthropic` gives you no embedder

Anthropic has no embedding API, so `with_anthropic` wires
`DeterministicEmbeddingProvider` — an **FNV-1a hash**, not a model. The crate warns at
runtime: *"recall is structural, NOT semantic."*

Everything keeps working. It just stops being smart: semantically similar text will not
be found, so "how do I get paid" will not match a passage about invoicing. That is the
worst way for a system to degrade, because nothing errors.

If you want Claude for reasoning **and** real semantic recall, pair it yourself:

```rust ignore
Memory::open(path)
    .with_llm(anthropic_provider)   // Claude for extraction
    .with_embedder(real_embedder)   // something that actually embeds
    .await?
```

Runnable: `cargo run --example hosted_providers` — the only example that costs money.

### Changing your embedding model later

Embeddings are only comparable to other embeddings from the **same** model. Change model
and every stored vector becomes noise relative to new queries — silently. Two operations
exist and picking the wrong one is expensive:

| operation | use it when |
|---|---|
| `backfill_episode_embeddings(batch)` | rows have **no** vector yet (you added an embedder to an existing corpus) |
| `reembed_all_episode_embeddings(batch)` | the **model changed** — the vectors are not missing, they are wrong |

⚠️ **Reaching for `backfill` after a model change is the trap.** Nothing is missing a
vector, so it correctly finds no work, reports success, and leaves the entire corpus
stale. Both are idempotent, so retries need no bookkeeping.

Runnable: `cargo run --example changing_embedding_model`.

The Tier 1 shortcuts (`with_ollama`, `with_openai`, `with_anthropic`) apply token tracking internally — no opt-in required.

To override the bundled cost rates table:

```rust
let mem = Memory::open("./agent.db")
    .with_provider_rates_path("./my-rates.toml")  // override bundled rates
    .with_llm(Arc::new(my_llm))
    .with_token_tracking("openai", "gpt-4o-mini")
    .with_embedder(Arc::new(my_embedder))
    .await?;
```

Full observability surface — emitted metrics, label schema, OTel/OTLP export, cardinality discipline, dashboard examples — documented in [observability.md](observability.md).

---
