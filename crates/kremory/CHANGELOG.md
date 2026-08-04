# Changelog

All notable changes to the `kremory` crate. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this crate uses semver.

## [0.6.0] - 2026-08-05

**Minor (breaking): out-of-the-box recall quality goes from ~25.7% to ~86%+ because
`content-search` and the dense episode arm are now DEFAULT features.** Every prior
published version shipped `default = []`, so `cargo add kremory` silently produced the
weakest possible build. If you pinned `default-features = false`, nothing changes for you.

Also lands the v1 durability sweep: sink events and durable-write counters no longer
describe rows a rollback erased, and two migration crash-resume defects are fixed.

### Changed (breaking)
- **`content-search` is a DEFAULT feature (ADR-078).** Measured 25.7% -> 86.2% recall
  out of the box. This is the headline reason to upgrade.
- **Dense episode arm ON by default.**
- **Contradiction detection default flipped OFF then back ON** with a rewritten prompt
  after it was found to destroy set-valued facts (TD-167/TD-165).

### Added
- **`RetrievedContext::is_content_passage()` + `CONTENT_PASSAGE_TYPE_NAME`.** Since
  `content-search` became a default, `recall(..).raw()` returns a HETEROGENEOUS list —
  content passages interleaved with graph entities — and a passage's `entity_id` is an
  EPISODE id, not an entity id. Passing it to an entity-scoped call fails with
  `no entity '<n>' in namespace`. `entity_type_id` cannot disambiguate either (a passage
  carries `0`, which is also the unknown-entity catch-all). Filter with
  `.filter(|r| !r.is_content_passage())`. Mirrored on the napi surface as the
  `isContentPassage` field.
- **`IngestStatus::Skipped`** — a terminal status for `skip_extraction` ingests, which
  previously stayed `Pending` forever and surfaced to callers as `WaitTimeout`.

### Fixed
- **Sink events + durable-write counters fired INSIDE the ingest transaction**, so a
  rolled-back ingest told consumers about entities and edges that do not exist, and never
  corrected itself. Now buffered and flushed only after the durable commit.
- **`forget().by_source_id()` was a 4-statement cascade with no transaction** — a failure
  mid-cascade left content stored but permanently unfindable.
- **Two migration crash-resume defects**: an un-dropped `sqlite_master` cursor that made
  `migrate_004`'s resume path fail with `database table is locked` (only ever on the
  resume path, which is why it was invisible), and `migrate_006`'s idempotency gate
  shadowing its own `episodic_edges` recovery gate — leaving `open()` permanently broken
  after a crash in the second half.
- **Migration resume is now CONTENT-based**, comparing against the immutable `_bak_`
  snapshot rather than trusting that a scratch table exists — an incomplete scratch was
  being renamed over live data.
- **A single bad candidate pair no longer aborts the whole `acronym_nickname_recall`
  dream pass**, discarding every other adjudicated merge.
- **`NuExtractEntitiesOnly` token cost was attributed to `operation="unclassified"`.**

### Known issues
- **A dream pass can still fail intermittently (~1 run in 5)** when a merge candidate's
  endpoint is absent at apply time. The failure is now isolated to that pair and counted
  (`kremory.identity.merge_apply_failed_total{site="site5"}`) instead of aborting the
  pass, but the root cause is not yet understood.

## [0.5.0] - 2026-07-14

Minor: the recall response becomes LLM-consumable (returns connected facts),
plus the recall-findability + vector-index bug fixes surfaced by dogfooding the
MCP server as kremory's first consumer. Backward-compatible types (additive on
`#[non_exhaustive]` `RetrievedContext`); the default recall render now emits
fact sentences, and `RetrievedContext.facts` / napi `RetrievedContext.facts` are
new. (Rolls up the prepared-but-unpublished 0.4.1 patch.)

### Added
- **`recall` returns connected facts (TD-116, ADR-074)** — each `RetrievedContext`
  now carries `facts: Vec<RetrievedFact>`, the LLM-consumable knowledge: a
  natural-language `fact` string ("Grace Hopper invented the compiler") + the
  structured triple + BOTH bi-temporal clocks (`valid_at`/`invalid_at` +
  `recorded_at`/`expired_at`) + confidence + source episode ids + score — richer
  than any surveyed peer (Graphiti/Zep/Mem0). Surfaced across the Rust facade
  (`.raw()` + templates), napi (`RetrievedContext.facts` / `RetrievedFact`), and
  the MCP `format:structured` output. The default `TemporalFacts` render now emits
  the fact sentences instead of the entity name + type-label summary.

### Fixed
- **Caller-pinned facts are now recall-findable (TD-113)** — `remember(...).with_facts(...).skip_extraction()`
  previously pinned facts that `recall` could not return (0 results), undercutting the
  "no second LLM" path. Pinned entities are now stamped at write time via three channels,
  all LLM-free: FTS name (`properties["name"]`), a vector embedding of the literal name, and
  an episodic edge for source attribution (without which a found entity renders empty under
  the default `TemporalFacts` template).
- **DiskANN vector index revived (TD-115)** — the `entities` embedding column was declared as
  a bare `BLOB` by earlier rebuild migrations, which `libsql_vector_idx` rejects; the failed
  index-create was swallowed, so *all* vector recall had silently fallen back to brute-force
  (correct but O(n) — no ANN speedup). New idempotent migration `migrate_023` rebuilds the
  column as `F32_BLOB(dim)` and creates the index loudly; the two prior swallowed index-create
  sites now emit a `kremory.search.vector_index_create_failed` counter + warning.
- **Filtered-ANN per-namespace recall shortfall (TD-114)** — `vector_top_k` post-filters
  `group_id`, so a small-fraction namespace could be under-filled after the filter. The index
  path now over-fetches scaled by estimated namespace selectivity (capped) and emits
  `kremory.search.namespace_recall_shortfall_total` on a true ANN-horizon shortfall.

## [0.4.0] - 2026-07-12

Minor bump on the pre-1.0 lane (breaking changes permitted). Adds the reversible
graph-mutation surface (ADR-073) and content recall (ADR-072 seq1), and fixes a
fact-extraction production bug on the `ner` build.

### Added
- **Reversible graph mutations (ADR-073)** — `unmerge`, `edit_entity` (rename / re-type
  with FK + edge + community propagation), `delete_entity` (cascade), and their inverses
  `undo_entity_edit` / `undo_delete_entity` / undo-of-unmerge, plus `list_mutations`,
  `mutation_history`, and NOGOOD recording so a reversed merge is not re-proposed. Every op
  returns an honest outcome struct (repointed / removed / restored counts, `already_undone`
  idempotency). Mirrored 1:1 on the `kremory-napi` surface.
- **Content recall (ADR-072 seq1)** — `.content()` terminal on the recall builder performs
  FTS5/BM25 full-text search over raw episode text (migrate_022 `episodes_fts` virtual table
  + purge cascade). Gated behind the `content-search` feature.

### Changed
- **`dream()` consolidation ops default ON** — community detection, cross-episode merge,
  archival, and the supersession sweep all run by default; tune via `DreamOpts` +
  `RememberInto::with_opts(...)`. `DreamSummary` reports a `would-merge` vs `did-merge`
  split and per-op `consolidationOpsRan` flags.
- **napi parity** — the reversibility + dream surfaces are exposed on `@kgentic/kremory-node`
  with the same honest return shapes as the Rust facade.

### Fixed
- **Fact extraction was a silent no-op under `--features ner` / `--all-features`.**
  `Engine::ingest()` special-cased `#[cfg(feature = "ner")]` to always dispatch through a
  bare, LLM-less `GlinerExtractor` (`facts: vec![]`), discarding the builder-resolved
  extractor (`IntegerId` default / `GlinerLlm` / `Custom` BYOE) — so every fact-producing
  extraction via `ingest()` dropped all relationships on the `ner` build (~100 commits).
  Now always dispatches through the configured extractor. A regression guard
  (`engine_ingest_wrapper_dispatches_configured_extractor_persists_facts`) asserts
  `facts >= 1` through `ingest()` and is ungated so it runs in both the default and `ner`
  test matrices.

## [0.3.2] - 2026-06-24

### Fixed
- **Duplicate episodic presence edge** — an entity that was both extracted *and* the
  object of a fact received two `episodic_edges` rows for the same `(episode, entity)`
  (one `role="mention"`, one `role="object"`), so `recall` rendered the same source
  episode twice (observed as a fact appearing twice in retrieved context). The `role`
  tag is not read on the recall path, so presence is one fact per `(episode, entity)`.

### Changed
- **Migration 017** installs `UNIQUE(episode_id, entity_id, entity_group_id)` on
  `episodic_edges` (dedup-first, idempotent — safe on existing 0.3.x databases) and
  `insert_episodic_edge` now uses `INSERT OR IGNORE`, making "an entity appears in an
  episode at most once" a structural invariant. New counter
  `kremory.ingest.episodic_edge_dup_suppressed_total` surfaces any suppressed duplicate
  (≈0 in normal operation). Entity-merge edge remapping handles the new constraint.

See ADR-055 for the full rationale.

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
