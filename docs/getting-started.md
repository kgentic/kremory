# Getting Started

> **v0.7** — This guide was written from a real, captured transcript: a fresh
> directory outside this repo, `cargo add kremory` against the **published**
> crate on crates.io (not this repo's source), and a program written using
> only what `cargo doc` / [docs.rs](https://docs.rs/kremory) show — the same
> path a brand-new user takes. Every command and every line of output below
> is real, not smoothed over. See [the API reference](./api/index.md) for the full reference
> once you're past this page.

## Prerequisites

- Rust + Cargo (this was run against `rustc 1.93.1` / `cargo 1.93.1`; kremory's
  stated MSRV is 1.86).
- For the worked example below specifically (not for kremory in general —
  kremory lets you bring your own LLM and embedder, see [Setup](./api/setup.md)): a local
  [Ollama](https://ollama.com) server with two models pulled:

  ```sh
  ollama pull gemma4:e4b
  ollama pull nomic-embed-text
  ```

  If you'd rather use OpenAI, Anthropic, or your own provider, see
  [Other ways to open a `Memory`](#3-other-ways-to-open-a-memory) below — you
  don't need Ollama to use kremory.

## 1. Create a project and add the crate

```sh
$ cargo init --name kremory_quickstart
    Creating binary (application) package
$ cargo add kremory
    Updating crates.io index
      Adding kremory v0.7.0 to dependencies
             Features:
             + content-search
             - embeddings
             - llm-integration
             - llm-smoke
             - ner
             - otel
             - rerank
             - test-utils
             - trace
             - unstable-graph
             - unstable-tags
```

Two things worth noting straight from that output, verified directly (not
assumed) as part of writing this guide:

- **`content-search` is ON by default.** `cargo add` prints it with a `+`,
  every other optional feature with a `-`. A bare `cargo add kremory` gives
  you the fused BM25/FTS5 + dense-episode recall arms out of the box — you do
  not need to opt in to get kremory's real recall quality (see
  [Feature flags](./api/feature-flags.md) for what each flag does).
- **`Cargo.toml` resolves to the real published crate**, not this repo:

  ```toml
  [dependencies]
  kremory = "0.8.0"
  ```

  and `Cargo.lock` confirms the source:

  ```
  name = "kremory"
  version = "0.8.0"
  source = "registry+https://github.com/rust-lang/crates.io-index"
  ```

Add the runtime + error-handling crates the example below needs (kremory's
API is `async` and returns its own error type):

```sh
$ cargo add anyhow tokio --features tokio/full
```

## 2. The smallest working example

This is genuinely the *first* thing that compiled and ran for this guide — it
was written by reading `cargo doc`'s rendered rustdoc for the crate root and
the `kremory::facade` module (both mirrored on
[docs.rs](https://docs.rs/kremory/0.7.0/kremory/)), not by copying this
repo's own examples:

```rust
use kremory::{Memory, Namespace};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Tier 1 "just works" path: Memory::with_ollama assumes Ollama running
    // at http://localhost:11434 with gemma4:e4b (chat) + nomic-embed-text
    // (embeddings) pulled — see "Other ways to open a Memory" below for
    // OpenAI / Anthropic / a fully custom chat provider + embedder.
    let mem = Memory::with_ollama("./agent.db").await?;

    let ns = Namespace::new("agent");

    mem.remember("User prefers concise replies")
        .in_namespace(ns.clone())
        .await?;

    let context: String = mem
        .recall("what does the user prefer?")
        .in_namespace(ns)
        .await?;

    println!("--- recalled context ---\n{context}");

    Ok(())
}
```

Run it:

```sh
$ cargo run
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.16s
     Running `target/debug/kremory_quickstart`
--- recalled context ---
Episode #1 (valid_at=2026-09-07T15:48:25.700966+00:00) — User prefers concise replies

user prefers concise replies (valid_at=2026-09-07T15:48:25.700966+00:00)
```

That's it — no path dependency, no workspace override, no code from this
repo other than the crate you just added. It compiled on the first try and
ran against a real local model with zero extra wiring.

### A real friction point: local-model output varies run to run

Re-running the exact same program (deleting `agent.db` between runs, same
two sentences, same local model) produced different recalled text each time
— for example, one run rendered the recalled entity as a generic `Entity`
catch-all rather than a named fact:

```
--- recalled context ---
Episode #1 (valid_at=2026-09-07T15:44:19.888110+00:00) — User prefers concise replies

user (valid_at=2026-09-07T15:44:39.731239+00:00) — Entity
```

**This is expected, not a bug** — `Memory::with_ollama`'s own doc comment
states the benchmark this default pairing was chosen against:
`gemma4:e4b` (reasoning disabled) gives **F1 84% / recall 90%** on kremory's
own extraction benchmark, not 100%. A small local model will occasionally
extract a fact more (or less) precisely than another run of the identical
input. If you need more consistent extraction quality, either use a larger
model (OpenAI/Anthropic, or Ollama's `qwen2.5:7b` costs recall per the same
doc comment) or don't be surprised by run-to-run variance with a 4B local
model — it's the tradeoff you're making for zero-network, zero-API-key
local-first operation.

## 3. Other ways to open a `Memory`

`Memory::with_ollama(path)` above is one of several Tier-1 "just works"
shortcuts. The others, discovered the same way (rustdoc), and lightly
verified:

- **`Memory::auto(path)`** — env-detects a provider: `$OLLAMA_HOST` →
  `$OPENAI_API_KEY` → `$ANTHROPIC_API_KEY` → `Err`. Unlike `with_ollama`,
  this does **not** default to `localhost:11434` on its own — `$OLLAMA_HOST`
  must actually be *set* (confirmed by running it with a cleared
  environment):

  ```
  Error: no provider configured: Set OLLAMA_HOST, OPENAI_API_KEY, or ANTHROPIC_API_KEY, or use Memory::open() builder with explicit providers
  ```

  Set `OLLAMA_HOST=http://localhost:11434` (or an OpenAI/Anthropic key) and
  `Memory::auto` behaves identically to `Memory::with_ollama` above.
- **`Memory::with_openai(path)`** / **`Memory::with_anthropic(path)`** —
  same shape, gated on `$OPENAI_API_KEY` / `$ANTHROPIC_API_KEY`.
- **`Memory::open(path).with_llm(...).with_embedder(...).await?`** — Tier 2,
  fully custom BYOM (bring your own chat provider) plus your own embedder.
  This is what you want for a production deployment or a
  non-Ollama/OpenAI/Anthropic backend. See [Setup](./api/setup.md) for the
  full builder walkthrough — it needs more setup than this guide's minimal
  path, which is why it isn't the first thing shown here.

## Next steps

- [The API reference](./api/index.md) — the full reference (builder customization,
  namespaces, dream/consolidation, undo, error handling, feature flags).
- [`deployment.md`](./deployment.md) — the three constraints that shape a real
  deployment: one connection per handle (no pool), an in-process-only write lock
  (cross-process is untested), and ingest measured in **seconds** not
  milliseconds, so writes must come off the request path.
- [`observability.md`](./observability.md) — metrics + tracing if you want
  to see what kremory is doing under the hood.
- [`error-handling-policy.md`](./error-handling-policy.md) — what kremory
  returns as an `Err` vs. what it treats as an invariant violation
  (`panic!`).
- The Node binding (`@kgentic-ai/kremory-node`) is **not yet published** —
  see [the Node binding](./api/node-binding.md) if you're evaluating it, but `npm install`
  will not work today.
