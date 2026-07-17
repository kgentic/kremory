# Contributing to kremory

Thanks for considering a contribution. kremory is pre-1.0 and evolving quickly — please open an
issue to discuss non-trivial changes (new public API, new cargo feature, schema/migration
changes) before writing code, so the design gets a look before the diff does.

## Project layout

A Cargo workspace, five crates:

| Crate | What it is |
|---|---|
| `crates/kremory` | The core library — bi-temporal graph engine, extraction, recall, dream consolidation, reversible mutations. Publishes to crates.io. |
| `crates/kremory-napi` | Node.js binding (napi-rs) mirroring the `Memory` facade in camelCase. Not yet published to npm. |
| `crates/kremory-mcp` | MCP (Model Context Protocol) server wrapping `Memory` as JSON-RPC tools. Not yet published as an installable package. |
| `crates/kremory-eval` | Extraction/recall precision benchmarking harness (used by `scripts/model-benchmark/`). |
| `crates/kremory-admin` | Internal admin/inspection tooling. |

## Dev setup

1. **Rust toolchain** — MSRV is **1.86** (`rustup toolchain install 1.86`; the workspace
   `Cargo.toml` declares `rust-version = "1.86"`, required for native `async fn` trait-object
   dyn-compatibility — ADR-040). `rustup component add clippy rustfmt`.
2. **Clone + build**:
   ```bash
   git clone https://github.com/kgentic/kremory
   cd kremory
   cargo build --workspace
   ```
3. **Test runner** — install [`cargo-nextest`](https://nexte.st/) (`cargo install cargo-nextest
   --locked`). It's the recommended local runner: on this workspace it's been measured at ~16×
   faster than `cargo test` for the full suite (parallel binary pool vs. serial). Note nextest
   does **not** run doctests — run those separately.
   ```bash
   cargo nextest run --workspace --all-features
   cargo test --doc -p kremory
   ```
4. **Optional: Ollama** for the two live-model test tiers below (`ollama pull gemma4:e4b &&
   ollama pull nomic-embed-text`).

## The 3-tier test pyramid

Most of the suite runs with **zero external dependencies** (mocked LLM/embedder). Two extra
tiers opt into real models:

| Tier | Command | Needs Ollama? | Roughly |
|---|---|---|---|
| **Default** | `cargo nextest run --workspace --all-features` | No | ~35s, ~1,200+ tests, mocked providers |
| **`llm-smoke`** | `cargo test -p kremory --features llm-smoke -- --ignored` | No (replays a checked-in VCR cassette by default) | ~2 min, one golden-path journey through the public `Memory` facade |
| **`llm-integration`** | `cargo test -p kremory --features llm-integration -- --ignored` | **Yes**, live Ollama | ~30-60 min, full real-LLM integration suite (source of truth for model behaviour: `crates/kremory/tests/llm_integration.rs`) |

For a PR: the default tier must pass. If your change touches extraction, recall, or dream-phase
LLM prompts, also run `llm-smoke` (and `llm-integration` if you have Ollama available) before
opening the PR, and say in the PR description which tiers you ran.

## Quality gate (required before every commit)

All four must be clean — no exceptions, no band-aids:

```bash
cargo check --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --check
cargo nextest run --workspace --all-features   # + cargo test --doc -p kremory
```

Use `--all-features` (not a single named feature) for clippy — cfg-gated code behind one
feature is invisible to clippy runs that don't enable it, so a partial-feature clippy pass can
miss real lint failures in feature-gated modules.

Also run `cargo build --example quickstart -p kremory` if you touched the public facade —
that example is the drift guard for the README's Quickstart snippet.

**A failing check means fix the cause, not silence the symptom.** Concretely, in this
codebase:

- **No `#[allow(clippy::...)]` in `src/`.** If clippy's `too_many_arguments` fires, restructure
  the function to take a params struct (see `RecallParams`, `ContentSearchParams`, etc. for the
  existing pattern) — don't suppress the lint. This is enforced by convention, not yet by CI (CI
  is currently disabled — see below), so please self-check.
- **No `unwrap()` / `expect()` in `src/`** — both are `deny`-level clippy lints on this crate
  (`crates/kremory/Cargo.toml` `[lints.clippy]`). Push validation to the API boundary, use a
  `Result`-returning path, or a documented poison-recovery pattern — not a suppressed panic.
- **A failing test means the test found a real bug** — fix the implementation or fix the test's
  premise; don't mark it `#[ignore]` to get green.
- **A clippy/fmt failure means fix the code**, not add an `#[allow]` or reformat exception.

If you disagree that a lint should fire on a specific line, raise it in the PR rather than
silently suppressing — there may be a real fix, or the lint config may need revisiting.

## A note on CI

GitHub Actions is currently disabled for this org (billing) — see the commented-out workflows
under `.github/workflows/`. That means **PR review is the only gate today**: run the quality
gate above locally before opening a PR, and expect a maintainer to re-run it before merging.

## PR flow

1. Fork (or branch, if you have write access) and make your change on a feature branch.
2. Keep commits scoped — one logical change per commit; a good commit message explains *why*,
   not just *what*.
3. Run the quality gate (above) before opening the PR.
4. Update the [CHANGELOG](crates/kremory/CHANGELOG.md) `Unreleased` section if your change is
   user-facing (release-please manages version bumps from this file).
5. If you touched the public `Memory` facade (a `pub fn` on `Memory`/`IngestRequest`/
   `RecallRequest`/`ForgetRequest`/`DreamRequest`, or a `pub` field on an exported struct), also
   update `crates/kremory-napi/` to keep the Node binding in sync — or add the symbol to
   `crates/kremory-napi/parity-skip.toml` with a reason if it's intentionally Rust-only.
6. Open the PR with: what changed, why, and which test tiers you ran.
7. Be responsive to review — this is a small team, expect direct/terse feedback.

## Reporting bugs / requesting features

Open a GitHub issue. For bugs, include your `kremory` version, a minimal repro if possible, and
which cargo features you had enabled. For security issues, see [SECURITY.md](SECURITY.md)
instead of a public issue.

## License

By contributing, you agree your contributions are licensed under the [Apache License 2.0](LICENSE),
the same license as the rest of the project.
