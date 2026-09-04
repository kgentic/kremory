---
title: 'kremory vs alternatives — feature comparison'
type: doc
status: active
created: '2026-05-22'
slug: docs/comparison-2026-05-22
tags:
  - kremory
  - comparison
  - competitive
  - codemem
  - mem0
  - letta
  - zep
  - graphiti
  - codebase-memory-mcp
refs:
  - id: rql/adr-001-engine-architecture-single-crate-apache2-2026-05-22
    rel: implements
  - id: rql/adr-002-byom-distribution-moat-2026-05-22
    rel: implements
  - id: rql/adr-003-bitemporal-audit-compliance-2026-05-22
    rel: implements
  - id: rust-agent-memory-competitor-landscape-synthesis-2026-05-22
    rel: informed_by
---

# kremory vs Alternatives — Feature Comparison

> Data current as of 2026-05-22. All competitor cells verified against primary sources (GitHub repos, crates.io, npm, PyPI, official docs). kremory cells reflect the locked ADR chain (v3 licensing ADR + ADR-001 through ADR-005 + architecture spec). Sources cited per cell.
>
> ⚠️ **PARTIALLY STALE (flagged 2026-09-04, not corrected wholesale).** This doc predates kremory's
> first public release by ~3.5 months. The **kremory-only cells below marked ✏️ have been updated**
> against this repo's current state, which is directly verifiable with no external research. The
> **competitor cells (stars, tool counts, integration counts for codemem / codebase-memory-mcp /
> Mem0 / Letta / Zep) have NOT been re-verified** — no live source access was available at update
> time, and per this project's own verify-before-stating discipline, an unverified guess is worse
> than an honest staleness flag. Treat every non-✏️ cell as a claim from 2026-05-22, not today.

---

## Comparison Matrix

| Dimension | **kremory** | cogniplex/codemem | DeusData/codebase-memory-mcp | Mem0 OSS | Letta OSS | Zep/Graphiti |
|---|---|---|---|---|---|---|
| **Language** | Pure Rust | Pure Rust (93% bytes) [codemem §1] | Pure C (95% bytes) [store.c verified] | Python | Python | Python |
| **License** | Apache-2.0 [ADR-v3] | Apache-2.0 [codemem Cargo.toml] | MIT [LICENSE: "Copyright (c) 2025 DeusData"] | Apache-2.0 [licensing-revisit §3] | Apache-2.0 [licensing-revisit §3] | Apache-2.0 (Graphiti) [licensing-revisit §3] |
| **Distribution** | crates.io (`cargo add kremory`) | crates.io (`cargo add codemem`) | install script / binary releases [README] | PyPI (`pip install mem0ai`) | PyPI (`pip install letta`) | PyPI (`pip install graphiti-core`) |
| **Storage substrate** | libSQL (Turso-compatible; embedded or remote) [ADR-001] | rusqlite 0.39 bundled SQLite — local only [codemem Cargo.toml §3.2] | Pure C + SQLite (`cbm_store_t` opaque handle) [src/store/store.c verified] | Postgres / hosted cloud [licensing-revisit §3] | SQLite + cloud [licensing-revisit §3] | Postgres + Neo4j / hosted cloud [licensing-revisit §3] |
| **Graph implementation** | Home-grown bi-temporal graph on libSQL [ADR-001] | petgraph 0.8 persisted to SQLite [codemem §3.3] | SQLite tables only — no graph library; semantic graph represented as edge rows [store.c] | Python networkx-style / cloud graph [licensing-revisit §3] | Python native / cloud [licensing-revisit] | Custom Python graph engine [licensing-revisit §3] |
| **Embedding model** | BYOM — `Arc<dyn EmbeddingProvider>` required; no model bundled [ADR-002] | Bundles Candle + BAAI/bge-base-en-v1.5 (~440MB) [codemem Cargo.toml: candle-core, hf-hub] | Bundled embedding via tree-sitter semantic tokens — not a neural embedder [README] | OpenAI embeddings + Qdrant / cloud [licensing-revisit] | OpenAI / configurable [licensing-revisit] | OpenAI / configurable [licensing-revisit] |
| **crates.io publishable** | Yes — engine crate is ~1MB (BYOM, no model) [ADR-002] | Yes — `codemem` publishes on crates.io | N/A (not a Rust crate) | N/A (Python) | N/A (Python) | N/A (Python) |
| **Time model** | Two-axis: `recorded_at` (TX time, immutable) + `valid_from`/`valid_to` (valid time, mutable) [ADR-003] | One axis: `valid_from`/`valid_to` on nodes/edges via migrations 003+015; no `recorded_at` [codemem §3.6] | None — no temporal model in storage layer [README, store.c verified] | None — vector recency only [licensing-revisit §3] | None [licensing-revisit §3] | Partial — entity/fact timestamps, one clock [licensing-revisit §3] |
| **Contradiction resolution** | Active engine: `ContradictionResolution` enum (Superseded / Merged / Forked / Ignored); `valid_to` set on invalidated facts [ADR-003] | Labels only: `Contradicts`/`InvalidatedBy`/`Supersedes` edge types in `RelationshipType` enum; no resolver algorithm found in source [codemem §3.7] | None [README, store.c] | None [licensing-revisit §3] | None [licensing-revisit §3] | Partial: supersession edges; no active resolver [licensing-revisit §3] |
| **MCP tool count** | ✏️ 5 tools (remember / recall / dream / list_mutations / undo) — shipped in this repo (`crates/kremory-mcp`), not yet published as an installable package [see README "Node.js / MCP"] | 32 tools via JSON-RPC stdio or HTTP [codemem §3.9, unverified since 2026-05-22] | 14 tools [README: "14 MCP tools", unverified since 2026-05-22] | Yes — Python SDK + cloud API | Yes — Python SDK + REST | Yes — Graphiti Python SDK |
| **Tree-sitter grammar support** | None (not a code-analysis tool) | No tree-sitter (code understanding via LLM extraction) | 155 vendored tree-sitter grammars [README: "155 tree-sitter grammars"] | None | None | None |
| **Embedded (no server process)** | Yes — libSQL file, no server process required [ADR-001, ADR-002] | Yes — rusqlite file, no server process | Yes — pure C binary, no server | No — requires Postgres/cloud | No — requires external services | No — requires Postgres/Neo4j |
| **Agent integrations** | ✏️ Via kremory-mcp (5 tools shipped, package not yet published — see above) | ~10 integrations (Claude Code, Cursor, Windsurf, Copilot, Zed, others) [codemem README, unverified since 2026-05-22] | 11 agent integrations [README: "11 agent integrations", unverified since 2026-05-22] | Many — official integrations page | Many — official integrations page | Many via Graphiti SDK |
| **GitHub stars** | ✏️ Not re-verified 2026-09-04 (no live GitHub access at update time — do not quote the "0, pre-launch" figure, it predates the public release) | 11 [GitHub API, 2026-05-22, unverified since] | 2,502 [GitHub API, 2026-05-22, unverified since] | ~20K+ | ~10K+ | ~5K+ (Graphiti) |
| **Active development** | ✏️ Active — 0.7.0 released 2026-09-04, LoCoMo benchmark added (see README) | Active — v0.18.0, weekly releases [codemem §4, unverified since 2026-05-22] | Active [README arXiv 2603.27277, unverified since 2026-05-22] | Active, Series A funded | Active, seed funded | Active, YC funded |

---

## When to Choose kremory

**kremory is the right choice when:**

1. **You need an embeddable agent memory library — not a service.** kremory is a Rust crate you `cargo add` into your application. No server process. No container. No subscription required to ship a working product. The SQLite of agent memory: embed it, ship it, move on.

2. **You are building a general-purpose agent system — not a coding assistant.** kremory stores facts about any domain (people, decisions, events, products, relationships). codemem and codebase-memory-mcp are scoped to code artifacts. kremory has no such scope constraint.

3. **You need audit-grade temporal correctness.** kremory tracks two independent time axes: when facts were recorded (immutable) and when facts were true in the world (mutable via contradiction resolution). This is the legal/compliance/healthcare requirement. No other listed tool provides two-clock bi-temporal model with an active contradiction resolver.

4. **You need BYOM / cost control / privacy.** kremory never calls an embedding or LLM endpoint on your behalf. Your API keys stay yours. Your inference costs stay on your bill. Your data never leaves your infrastructure. codemem bundles a ~440MB model and calls it automatically. kremory does not.

5. **You are writing Rust and want a native, async, zero-overhead integration.** Pure Rust, `async_trait`-compatible, `Send + Sync` trait objects throughout. No FFI, no subprocess, no network hop in the hot path.

---

## When to Choose Alternatives

**cogniplex/codemem — choose when:**
- Your agent system is a coding assistant (Claude Code plugin, Cursor extension, Windsurf) and you need deep code-graph memory with IDE integrations ready today
- You need 32 pre-built MCP tools NOW (kremory-mcp ships at v0.2.0)
- Your use case is "understand this codebase" not "remember facts about a user or domain"

**DeusData/codebase-memory-mcp — choose when:**
- You need fast, pure-C, zero-Rust codebase indexing with 155 language grammars
- You want to index the Linux kernel in 3 minutes
- You are working in C/C++ or need a minimal binary without a Rust toolchain
- Note: "team sharing = commit compressed file to git" — there is no hosted sync

**Mem0 OSS / letta OSS / Zep/Graphiti — choose when:**
- You are writing Python and want a Python-native library
- You want a production-ready hosted service today (mem0.ai, letta.ai, getzep.com)
- You have a funded team and want vendor support
- You do not need embedded / offline / local-first operation
- Note: Mem0 v2 removed its graph layer; Letta is agent framework + memory; Graphiti is the graph-memory component of Zep

---

## Migration Paths

### Migrating from codemem to kremory

codemem stores memories in a rusqlite SQLite file. The schema uses `graph_id`, `entity_id`, and `edge_id` tables with temporal columns. Migration path:

1. Export codemem graph via `codemem export --format json` (or direct SQLite read)
2. Map codemem entities → kremory episodes via `memory::submit_episode`
3. kremory's contradiction resolver will re-derive temporal validity from episode content; codemem's `valid_from`/`valid_to` can be mapped to kremory's valid-time axis manually
4. kremory's `recorded_at` will reflect the migration timestamp (not the original codemem ingest time) — this is a known limitation of migration

### Migrating from codebase-memory-mcp to kremory (via codebase-memgraph-kremory fork)

The planned `codebase-memgraph-kremory` fork (kremory v0.4+ roadmap) will replace codebase-memory-mcp's C storage layer with kremory while preserving the 155 tree-sitter grammars, 14 MCP tools, and 11 agent integrations. Migration path for that fork is a drop-in replacement of the MCP server binary.

For direct migration (before the fork ships): codebase-memory-mcp stores data in a SQLite file via the `cbm_store_t` abstraction. Export via the `export_graph` MCP tool, then re-ingest into kremory as code-context episodes.

### Migrating from Mem0 to kremory

Mem0 exposes a REST export API. Consume the JSON output and re-ingest via `memory::submit_episode`. kremory's enrichment pipeline will re-extract entity relationships from episode content.

### Migrating from Zep/Graphiti to kremory

Graphiti exposes a Python client with `get_episodes` and `get_nodes`. Export via the Graphiti client, then re-ingest into kremory. Temporal data from Graphiti's single-clock model maps to kremory's `valid_from`/`valid_to` (valid-time axis); `recorded_at` will reflect migration time.

---

## Sources

All competitor data verified against primary sources (2026-05-22):

- `codemem`: `~/Documents/Projects/Ideas/kremory/.ai-docs/research/competitors/codemem-2026-05-22.md`
- `codebase-memory-mcp` LICENSE: `~/Documents/Projects/Ideas/oss/codebase-memory-mcp/LICENSE` (read directly, MIT confirmed)
- `codebase-memory-mcp` storage: `~/Documents/Projects/Ideas/oss/codebase-memory-mcp/src/store/store.c` (head read, `cbm_store_t` confirmed)
- `codebase-memory-mcp` README: `~/Documents/Projects/Ideas/oss/codebase-memory-mcp/README.md` (2,502 stars, 14 tools, 11 integrations, 155 grammars — verified)
- Mem0 / Letta / Zep: `.ai-docs/research/rqlm-licensing-revisit-research-2026-05-19.md`
- Synthesis: `.ai-docs/research/rust-agent-memory-competitor-landscape-synthesis-2026-05-22.md`
- Fork analysis: `.ai-docs/research/codebase-memory-mcp-fork-analysis-2026-05-22.md`
