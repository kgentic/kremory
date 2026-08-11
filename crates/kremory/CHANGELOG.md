# Changelog

All notable changes to the `kremory` crate. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this crate uses semver.

## [Unreleased]

### Fixed — dream alias resolution was inert on real corpora (TD-203)

On the shipped LoCoMo corpus, **42 of 42 alias candidates sat unresolved across ten
completed dream runs.** Re-running the same unchanged pass on a copy resolved 28 of them in
under a second. Four defects, all now closed:

- **Entity merge manufactured self-loops.** Re-pointing a fact whose *other* endpoint was
  the merge keeper produced `X pred X`, which asserts nothing — 41 live ones on that corpus,
  18 on the reserved `potential_alias` predicate, a shape disambiguation cannot even emit.
  Such facts are now expired by the merge, and **revived by `unmerge`** so the mutation stays
  reversible (ADR-073).
- **The dream pass that RESOLVES aliases ran before the two passes that CREATE them.**
  Anything produced by acronym/nickname recall or by canonicalization's merges waited for a
  *next* dream, which a one-dream-per-namespace caller never provides. A second deterministic
  sweep now runs after consolidation. `DreamSummary.aliases_resolved` therefore now counts
  both sweeps.
- **The pass was indistinguishable from a no-op.** It reported only merges+revokes, with
  every other outcome at `debug!`, so "examined 42 and kept them all" looked identical to
  "found nothing". It now emits one always-on summary line with `candidates_examined` and a
  per-outcome breakdown, warns on an alias referencing an absent entity, and trips a warning
  if candidates go in and nothing is judged.
- **The alias re-similarity read was namespace-unscoped** (`WHERE a.id = ? AND b.id = ?`)
  while the entity key is the composite `(id, group_id)`. On that corpus the join matched up
  to **100 rows** and took an arbitrary one. Harmless there only because same-named entities
  across namespaces happened to carry identical vectors; now scoped.

### Fixed — ingest overwrote entity embeddings across ALL namespaces (TD-206)

**Cross-namespace corruption of a live retrieval signal, on the default ingest path.**
`ingest_with.rs` set every extracted entity's embedding via a helper whose SQL matched
`WHERE id = ?` with **no `group_id`**, while the entity key is the composite `(id, group_id)`.
Ingesting an entity named `X` into one namespace therefore overwrote `X`'s embedding in every
other namespace holding that name — silently, and cross-tenant for a multi-tenant consumer.
**90 entity names exist in more than one namespace on the benchmark corpus.**

Now writes through a namespace-scoped `set_entity_embedding_in_group`; the unscoped helper is
compiled out of production builds entirely, so the mistake is unrepresentable rather than
discouraged by a doc comment.

### Added
- `TemporalGraph::set_entity_embedding_in_group` + `SetEntityEmbeddingParams` — the scoped
  embedding write. `set_entity_embedding` is now `#[cfg(any(test, feature = "test-utils"))]`.

## [0.6.0] - 2026-08-11

**Minor (breaking): out-of-the-box recall quality goes from ~25.7% to ~86%+ because
`content-search` and the dense episode arm are now DEFAULT features.** Every prior
published version shipped `default = []`, so `cargo add kremory` silently produced the
weakest possible build. If you pinned `default-features = false`, nothing changes for you.

Also lands the v1 durability sweep: sink events and durable-write counters no longer
describe rows a rollback erased, and two migration crash-resume defects are fixed.

Folded in since this entry was first drafted: `recall()` no longer leaks kremory's own
entity-disambiguation bookkeeping to consumers as invented facts (TD-197 — up to 142 of
199 benchmark questions were affected before the fix), a bi-temporal anchoring bug where
LLM-extracted facts ignored the caller's declared `published_at` (TD-181), and a fix for
entities mentioned before being formally extracted, which were previously unreachable by
both text and vector search. Also ships a breaking addition to `RawFact` for external
`EntityExtractor` implementations, a fully-public construction surface for
`RetrievedContext` / `RetrievedFact`, and two dependency security fixes.

### Changed (breaking)
- **`content-search` is a DEFAULT feature (ADR-078).** Measured 25.7% -> 86.2% recall
  out of the box. This is the headline reason to upgrade.
- **Dense episode arm ON by default.**
- **Contradiction detection default flipped OFF then back ON** with a rewritten prompt
  after it was found to destroy set-valued facts (TD-167/TD-165).
- **`RawFact` gains a new field, `valid_at: Option<String>` (TD-187/TD-192).** Breaking
  for external `EntityExtractor` implementations that construct `RawFact` via struct
  literal — it is `pub`, not `#[non_exhaustive]`. Default-identical for every built-in
  extraction path: the field is optional/`#[serde(default)]`, and an absent value falls
  back to the pre-existing `ref_time` behaviour.

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
- **`RetrievedContext::with_facts(...)` / `with_entity_type(id, name)`, plus
  `RetrievedFact::new(RetrievedFactNewParams)` (+ `with_invalid_at` / `with_expired_at`)
  (TD-199).** `RetrievedFact` previously had no public constructor at all, and
  `RetrievedContext`'s fluent-setter surface had no way to set `facts` or the
  entity-type pair — both `#[non_exhaustive]` types are now fully constructible outside
  the crate.
- **`kremory::memory::{RenderableContext, RenderableFact, RenderableSourceRef}` traits,
  and `render_entities` / `render_edge_summary` / `render_temporal_facts` are now `pub`
  (TD-198).** Lets a consumer render kremory's own Markdown context blocks over its own
  result type, not only `RetrievedContext`.
- **Extraction prompts are now temporally grounded, and facts can carry their own
  `valid_at` (TD-187).** When a caller sets `.published_at(...)`, the extraction prompt
  now includes that declared date, and an extracted fact's `valid_at` is used when the
  model supplies one (falling back to the episode's reference time otherwise) —
  previously every fact from one episode shared a single date regardless of what the
  source text actually said ("yesterday", "last March", etc.).

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
- **`recall()` no longer serves kremory's own entity-disambiguation bookkeeping to
  consumers as a fact (TD-197).** The reserved `potential_alias` meta-edge predicate —
  used internally to propose entity merges — was never filtered out of either the
  connected-facts projection or the dense fact-search arm. Measured on a 199-question
  benchmark: 142 questions were leaking it before the fix, 0 after.
- **Facts produced by LLM extraction now honour the caller's declared `published_at`**
  instead of always anchoring to ingest wall-clock (TD-181) — previously only
  caller-supplied structured facts respected it, so the same document could be
  bi-temporally anchored two different ways depending on which code path produced a
  given fact.
- **Entities first seen as a forward reference — mentioned inside a fact before being
  formally extracted — are now findable.** They previously had neither a name (missed
  by full-text search) nor an embedding (missed by vector search); both are now
  backfilled after commit.
- **A silent data-loss bug in the structured-output JSON repair path is fixed
  (TD-192).** The prior JSON-repair dependency flattened any array it had to recover
  rather than parse cleanly, silently dropping every element after the first while still
  reporting success. Replaced with a dependency (`jsonrepair`) that fails loudly instead
  of silently truncating; not observed to have fired on the benchmark corpus this was
  found against.

### Security
- **2 RUSTSEC advisories fixed** via targeted dependency bumps: `crossbeam-epoch`
  0.9.18 -> 0.9.20 (RUSTSEC-2026-0204), `quinn-proto` 0.11.14 -> 0.11.16
  (RUSTSEC-2026-0185). 4 remaining advisories (all `rustls-webpki`, reached only through
  a pinned `libsql = "=0.9.30"`) are documented as unreachable in kremory's usage — a
  regression test fails the build if any remote-libsql API is ever referenced — pending
  a non-prerelease upstream `libsql`.

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
