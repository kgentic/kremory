# Upgrade guide


### v0.0.x → v0.1.0

No existing consumers to migrate — v0.1.0 is the first public release.

If you used an internal snapshot (pre-tag), the changes are:

| Before | After |
|---|---|
| `WorkspaceScope::new("id")` | `Namespace::new("id")` |
| `WorkspaceScope::with_thread("id", "t")` | `Namespace::new("id").with_thread("t")` |
| `MemoryHandle::open(path, emb)` | `Memory::auto(path).await?` or `Memory::open(path).with_llm(...).with_embedder(...).await?` |
| `handle.submit_episode(...)` (9-arg) | `mem.remember(content).from_chat(id).in_namespace(ns).await?` |
| `handle.context_block(&scope, query)` | `mem.recall(query).in_namespace(ns).await?` |

The substrate free-functions (`kremory::memory::submit_episode`, etc.) remain public and
unchanged — if you depend on them directly, no migration is needed.

### v0.1.0 → v0.1.1 (shipped 2026-05-27)

Recall pipeline redesign. No public API changes; substrate-level behavior fixes only.

| Area | Change |
|---|---|
| RRF (Reciprocal Rank Fusion) | Constant `k = 60` now used for hybrid score blending |
| Episodic edges | Now authoritative; deprecated fallback paths removed |
| LightRAG stub-entity | Forward references insert stub entities at first mention; promoted on re-ingestion |
| First-mention snippet | `source_ref` carries first-encountered snippet, not most recent |
| `SourceKind::Episode` | Replaces deprecated `SourceKind::Document` |

No migration required for facade-tier consumers.

### v0.1.1 → v0.1.2 (shipped 2026-05-28)

LLM observability parity. Additive only — no breaking changes.

| API | Change |
|---|---|
| `MemoryBuilder::with_token_tracking(provider, model)` | Composable knob — wraps the already-configured LLM in `TokenTrackingChatProvider` with explicit labels. Supersedes the deprecated `with_llm_tracked(WithLlmTrackedParams, llm)` |
| `MemoryBuilder::with_provider_rates_path(PathBuf)` | **NEW** — override bundled cost-rates table |
| `MemoryBuilder::with_llm(llm)` | Unchanged — backward compatible (untracked path) |
| Tier 1 shortcuts (`with_ollama` etc.) | Apply token tracking internally — auto-emit metrics |
| `kremory::observability` module | **NEW** — re-exports `TokenTrackingChatProvider`, `ProviderRates`, `TelemetryConfig`, `init_telemetry`, etc. |
| `init_telemetry(TelemetryConfig)` | Real implementation (was stub). Returns `TelemetryHandle`. Behind `otel` cargo feature. |
| `otel` cargo feature | Off by default. When enabled: installs `tracing-subscriber` + `tracing-opentelemetry` + OTLP gRPC exporter |
| `error.type` span attribute | **NEW** — bounded to `{server_error, client_error, parse_error}`; tracking span attribute (not histogram label) |
| `kremory_core_tokens_total` counter | Extended with `operation="chat"` emissions (embed emissions unchanged) |
| `kremory_core_cost_usd_total` gauge | **NEW** — cumulative USD cost, f64 |
| `kremory_core_chat_duration_seconds` histogram | **NEW** — per-call duration, `status` label bounded to `ok`/`error` |

See [observability.md](observability.md) for full surface.

### v0.1.2 → v0.1.3 (shipped 2026-05-28)

Hygiene only. No public API changes. `cargo clippy --all-targets --all-features` now passes clean.

### v0.1.3 → v0.3.2 — what changed (high level)

The changes since v0.1.3 are overwhelmingly **additive** — facade-tier consumer code from v0.1.3
continues to compile and run. The notable surface + behaviour changes, at the API-surface level:

| Area | Change |
|---|---|
| **Dream consolidation** ([dream](../api/dream.md)) | The dream phase is now fully wired — reconciliation passes (type discovery, aliases, reclassify, consistency-check, canonicalize) + four graph-global consolidation ops (community detection, cross-episode merge, supersession sweep, fact archival). All default ON; cross-episode merge defaults to **Shadow**. `DreamSummary` gained honest per-op fields (the old `episodes_processed` / `edges_merged` fields never existed on the shipped struct — use the fields documented in [dream](../api/dream.md)). |
| **Reversibility** ([reversibility](../api/reversibility.md)) | **NEW** — `mem.mutation_history` / `list_mutations` (SEE), the unified `mem.undo(mutation_id)` dispatcher + per-kind `unmerge` / `undo_entity_edit` / `undo_delete_entity` / `undo_delete_fact` / `unsupersede` / `restore_archived_fact`, plus direct mutations `edit_entity` / `delete_entity` / `delete_fact` / `supersede`. Every destructive mutation is reversible. |
| **Content recall** ([recall](../api/recall.md)) | **NEW** — `mem.recall(q).content()` for BM25/FTS5 search over raw episode text, under the `content-search` feature — **now ON by default**, which also RRF-fuses the BM25 + dense episode arms into the ordinary `recall()` path. |
| **Namespace policy enforcement** | v0.1.4 declared policies; **v0.1.5+ ENFORCES them**: `dream()` / `forget()` / `supersede()` now return `Error::NamespacePolicyViolation` on `AppendOnly` namespaces. This is a behaviour change if you registered `AppendOnly` policies expecting the v0.1.4 declare-only semantics. |
| **Multi-namespace recall** | `recall(q).in_namespaces(&[...])` fans out across namespaces with cross-namespace RRF blending. |
| **Metadata filters** | `recall(q).filter_metadata(key, value)` / `.filter_metadata_in(key, &[..])` post-filter recall by episode metadata. |
| **Async extraction wait** | `mem.wait_for_processing(episode_id, timeout)` polls an episode to a terminal extraction state. |
| **Feature flags** ([feature flags](../api/feature-flags.md)) | The crate's `default` feature set is **`["content-search"]`** (2026-07-28 — it was previously empty); see [feature flags](../api/feature-flags.md). There is no separate `substrate` feature — the facade + substrate free-functions ship in the one default surface. |
| **Node/napi binding** ([node binding](../api/node-binding.md)) | The JS binding mirrors the surface in camelCase, including the undo + inspect surface. |

The substrate free-functions (`kremory::memory::submit_episode`, etc.) remain public and unchanged.

---
### v0.3.2 → v0.6.0

`0.4.0` and `0.5.0` are additive. `0.6.0` carries the only breaking item in this span, plus
the change that is the actual reason to upgrade.

| Area | Change |
|---|---|
| **`content-search` is now a DEFAULT feature** | Measured 25.7% → 86.2% recall out of the box. If you were opting in explicitly, you can drop the flag; if you were relying on the previous empty default set, recall quality changes substantially — for the better. |
| **Dense episode arm ON by default** | Recall now RRF-fuses the BM25 and dense episode arms into the ordinary `recall()` path. |
| **Contradiction detection** | Default flipped OFF and then back ON with a rewritten prompt, after the original was found to destroy set-valued facts. |
| **`RawFact` gained `valid_at: Option<String>`** | **BREAKING for external `EntityExtractor` implementations that build `RawFact` with a struct literal** — the type is `pub` and not `#[non_exhaustive]`. Every built-in extraction path is default-identical: the field is optional with `#[serde(default)]`, and an absent value falls back to the previous `ref_time` behaviour. Add `valid_at: None` to your literal, or switch to functional-update syntax. |

### v0.6.0 → v0.7.0

No breaking changes to the facade. One field removal that had no working callers, and a set of
correctness fixes worth knowing about because they change results you may have measured.

| Area | Change |
|---|---|
| **`SearchFilters.valid_after` / `valid_before` removed** | Declared since v0.1.0 and never wired to any query — a separate mechanism was built for facts. Zero non-test usages before removal. If you were setting them, they were doing nothing; use `recall().as_of(t)`. |
| **`recall().as_of()` now scopes episode and content search too** | Previously it filtered facts only, so a valid-time query could still return present-day episode text. Results change for any `as_of` query. |
| **Ingest no longer overwrites entity embeddings across namespaces** | A correctness fix; recall quality in multi-namespace deployments changes. |
| **Dream alias resolution was inert on real corpora** | Now functional, so `dream()` does more than it used to. |
| **Oversized documents no longer fail dense embedding silently** | New `split_for_embedding` helper; previously long inputs produced no embedding and no error. |
| **New `extraction_arm_budget_ms` builder knob** | Additive. |
| **`dream()` is no longer recommended as routine usage** | Documentation change, not behaviour — see [dream](../api/dream.md). |

`0.7.1` is packaging and public-surface hygiene only: no behaviour, API or schema change.

### v0.7.0 → v0.8.0

Two breaking changes, both mechanical.

**`dream()` now requires an explicit `.execute()` terminal.** `DreamRequest` no longer
implements `IntoFuture`, so awaiting the builder directly stops compiling. This matches every
other mutating request (`forget()`, `undo()`, `supersede()`), which already required a terminal
call.

```rust ignore
// Before (no longer compiles on 0.8.0)
mem.dream().await?;
// After
mem.dream().execute().await?;
```

**`NoEmb` / `WithEmb` renamed to `NoEmbedder` / `WithEmbedder`**, for consistency with the
existing `NoLlm` / `WithLlm` pair. This only affects code that names those marker types
explicitly rather than letting them be inferred through `MemoryBuilder`.

Also in this release, all additive: `Memory::with_ollama_at_model(url, model, path)`; `Debug`
implementations for `Memory` and `MemoryBuilder` (neither had one); prior-turn replay into the
extraction prompt so references resolve across conversation turns; and a fix for
`update_episode_metadata`, which overwrote every episode sharing a source id.

---
