# kremory-napi

Node.js binding for [kremory](https://github.com/kgentic/kremory), a bi-temporal knowledge-graph
memory engine for AI agents — remember, recall, and reversibly undo. Built on
[napi-rs](https://napi.rs/) against the same Rust core the `kremory` crate ships, mirroring the
Rust `Memory` facade in camelCase (`remember`, `recall`, `dream`, `undo`, and the rest).

## Status

> ⚠️ **`@kgentic-ai/kremory-node` is UNPUBLISHED.** `npm install @kgentic-ai/kremory-node` does
> not work today — there is no published package yet. Build the native module from source:
>
> ```sh
> pnpm install && pnpm build:debug
> ```

## Where to look next

- **[examples/](examples/)** — ten runnable, tested Node.js scripts covering the real API surface
  (namespaces, batch ingest, recall filters, dream/consolidation, reversibility, custom
  extractors/embedders). Start with [`examples/README.md`](examples/README.md).
- **[Node binding API reference](../../docs/api/node-binding.md)** — what's mirrored from Rust,
  what's Rust-only, and known gaps.
- **[`__test__/`](__test__/)** — the automated test suite (`pnpm test`).

## Known limitations

- Wiring a custom (JS-callback) embedder or LLM provider can hit a native teardown assertion on
  abrupt process exit (upstream napi-rs issue) — this gates the npm publish.
- Cross-platform builds are not yet produced; only macOS (`darwin-arm64`/`darwin-x64`) binaries
  exist locally today.
