# Changelog

All notable changes to the `kremory` crate. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this crate uses semver.

## [0.3.1] - 2026-06-24

### Fixed
- **`with_ollama` default model** changed from `qwen3.5:9b-mlx` (Apple-Silicon-only,
  thinking-capable → blew the inline 30s extraction budget) to **`gemma4:e4b` with
  reasoning disabled** (`think:false`) + `keep_alive("1h")`. Per kremory's own benchmark
  (`scripts/model-benchmark/`, M4 Max): gemma4:e4b+think:false is the best extraction model
  that fits the inline budget — F1 84 / recall 90% / ~16s slowest call. Reasoning *on* is
  both slower (44s/call) and lower quality (F1 75). Also fixes a latent keep-alive thrash
  (the path set no `keep_alive`, so the model unloaded between chunks).
- **README** corrected: install is `kremory = "0.3"` (no git dependency, no
  `[patch.crates-io]` stanza — 0.3.0's README wrongly claimed otherwise, which is why it was
  yanked); dead `../../` doc links → absolute GitHub URLs; stale `KREMORY_EXTRACTOR=hybrid` /
  `NuExtract` references → current builder knobs (`.with_gliner()`); model table refreshed
  with benchmarked figures.
- Removed an obsolete "ships scaffolding only / returns `NotImplemented`" doc note in
  `kremory::memory` (the public functions have been implemented since v0.2.x).

### Added
- `readme` field + `[package.metadata.docs.rs]` (builds with `otel` + `--cfg docsrs`).
- `scripts/model-benchmark/` — reproducible local-model benchmark (precision/recall/F1/
  per-call latency/size/thinking) any dev can run.

## [0.3.0] - 2026-06-24 [YANKED]

Yanked: shipped a README that described a git+patch install path that no longer applied, and
a default model (`qwen3.5:9b-mlx`) whose zero-config first run failed. The crate code is
sound; superseded by 0.3.1.

### Changed
- Dropped the `ChatProvider::model()` trait dependency; the model id now flows as plain data
  (builder → engine → consumers), so kremory builds against the published `autoagents-llm`
  with no `[patch.crates-io]` redirect.
