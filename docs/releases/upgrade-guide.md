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
| **Reversibility** ([reversibility](../api/reversibility.md)) | **NEW** — `mem.mutation_history` / `list_mutations` (SEE), the unified `mem.undo(mutation_id)` dispatcher + per-kind `unmerge` / `undo_entity_edit` / `undo_delete_entity` / `undo_delete_fact` / `unsupersede` / `restore_archived_fact`, plus direct mutations `edit_entity` / `delete_entity` / `delete_fact` / `supersede`. Every destructive mutation is reversible (ADR-073). |
| **Content recall** ([recall](../api/recall.md)) | **NEW** — `mem.recall(q).content()` for BM25/FTS5 search over raw episode text, under the `content-search` feature (ADR-072) — **now ON by default** (ADR-078), which also RRF-fuses the BM25 + dense episode arms into the ordinary `recall()` path. |
| **Namespace policy enforcement** | v0.1.4 declared policies; **v0.1.5+ ENFORCES them** (ADR-029b): `dream()` / `forget()` / `supersede()` now return `Error::NamespacePolicyViolation` on `AppendOnly` namespaces. This is a behaviour change if you registered `AppendOnly` policies expecting the v0.1.4 declare-only semantics. |
| **Multi-namespace recall** | `recall(q).in_namespaces(&[...])` fans out across namespaces with cross-namespace RRF blending. |
| **Metadata filters** | `recall(q).filter_metadata(key, value)` / `.filter_metadata_in(key, &[..])` post-filter recall by episode metadata. |
| **Async extraction wait** | `mem.wait_for_processing(episode_id, timeout)` polls an episode to a terminal extraction state (ADR-051). |
| **Feature flags** ([feature flags](../api/feature-flags.md)) | The crate's `default` feature set is **`["content-search"]`** (ADR-078, 2026-07-28 — it was previously empty); see [feature flags](../api/feature-flags.md). There is no separate `substrate` feature — the facade + substrate free-functions ship in the one default surface. |
| **Node/napi binding** ([node binding](../api/node-binding.md)) | The JS binding mirrors the surface in camelCase, including the undo + inspect surface. |

The substrate free-functions (`kremory::memory::submit_episode`, etc.) remain public and unchanged.

---
