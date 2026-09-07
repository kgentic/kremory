---
title: "Public docs + API-surface audit — Phase 1 findings (SC-003 table)"
type: findings
status: draft
created: 2026-09-07
spec: docs/specs/public-docs-and-api-surface-audit/spec.md
plan: docs/specs/public-docs-and-api-surface-audit/plan.md
tasks: docs/specs/public-docs-and-api-surface-audit/tasks.md
---

# Phase 1 — Semantic surface audit findings

Every row below was re-verified against the current tree (kremory `0.7.1`
unpublished / `0.7.0` published on crates.io) on 2026-09-07 by reading the cited
`file:line` directly — none of the 13 findings handed off from Phase 0, and
none of the two pre-seeded findings from `plan.md`, were transcribed without
re-checking. Two of the handed-off facts turned out to need correction; both
are called out inline where they occur (F5, and the general note on grouping
below).

This is the **Phase 1 work list** for Phase 2 (T2.1-T2.4). No code or doc file
outside this one was modified to produce it (read-only per this task's scope).

**Fixed/NOT-FIXED column**: every row is `PENDING (Phase 2)` — that disposition
is Phase 2's job per RULE-006/SC-004, not Phase 1's. The column is included now
so Phase 2 can fill it in directly on this table (RULE-006 Finding 7's
disambiguation: the SC-003 table *is* the Phase 2 findings table).

## Phase 2 disposition — summary (filled in 2026-09-07)

**All 39 rows: FIXED. Zero NOT-FIXED.** Every finding was resolvable at the layer
stated in Phase 1 (or, for F10/F17, per the two judgment calls pre-decided before
Phase 2 started — see below) using the real, current signature verified against
source, not the finding's own suggested code where that suggestion itself proved
wrong on compile (F5, F14 — see their Disposition notes).

- **F10** (`AuditSink`): pre-decided as a DOC fix — rewrote the example to use the
  harness's own `MySink` placeholder convention (already used unqualified
  elsewhere in this same doc) rather than inventing a shipped `AuditSink` type.
- **F17** (`graph_degree_weight`): pre-decided as a CODE fix — added
  `MemoryBuilder::with_graph_degree_weight(f32)` mirroring the `extraction_arm_budget_ms`
  precedent exactly (`PipelineConfigOverrides` field → `PipelineConfigBuilder`
  method → `MemoryBuilder` setter, no env var, matching that same precedent's
  scope). Required raising the napi parity-skip cap **90 → 91** for exactly ONE
  new entry (`MemoryBuilder::with_graph_degree_weight`, ADR-030 Form B, same
  rationale as its eleven now-siblings) — see `crates/kremory-napi/parity-skip.toml`
  and the raise-history comment block in `crates/kremory-napi/tests/api_parity.rs`.
  This is the risk register's R3 concern (skip-cap raises should stay rare and
  rationale'd) — it is the 4th such raise this repo has made and was made on the
  record with the file's own established convention, not silently.
- Added a dedicated regression test,
  `with_graph_degree_weight_reaches_live_search_config`
  (`crates/kremory/tests/it/td141_search_config_builder_seam.rs`), following that
  file's own established pattern for this exact class of setter (the TD-141
  `SearchConfig` builder-seam family), plus an assertion in that file's
  `unset_knobs_match_documented_defaults` that the default stays `0.05` (this
  axis is the one that's already LIVE — unlike its newer sibling axes, `0.0`
  here would silently disable a shipped, tested boost).

**Compile-audit harness before/after** (`cargo test --doc -p kremory-doc-examples`,
re-verified by direct re-run, not by eye): **10 passed / 24 total → 24 passed /
25 total.** The one remaining failure
(`docs_observability_md`, the "Reading metrics in your app" unit) is the
harness's own pre-existing, deliberately-undocumented-as-a-defect exception
(`metrics_exporter_prometheus` — see `crates/kremory-doc-examples/README.md`
"Known, deliberate gaps"): adding that third-party crate as a dependency purely
to make one illustrative line compile would be adding a dependency to pass a
test, not verifying a real capability. Not a regression, not one of the 39
findings, not fixed, and not counted against the 24/25.

**A genuine, load-bearing harness bug was found and fixed during this
re-verification** (not one of the 39 findings — a Phase 0 infra defect, found
because SC-004/RULE-006 discipline requires actually re-running the harness
rather than trusting doc edits by eye): `PRELUDE_REFRESH_LINES` in
`crates/kremory-doc-examples/generate_docs.py` bound the short placeholder names
`llm` / `emb` to BARE `MockChatProvider` / `NullEmbeddingProvider` values, but
the overwhelming majority of doc sections that use these two short names (as
opposed to `my_llm` / `my_embedder`, which several OTHER sections wrap explicitly
via `Arc::new(my_llm)`) call `.with_llm(llm)` / `.with_embedder(emb)` BARE,
assuming an already-`Arc`-wrapped provider — exactly the convention the harness's
own README documents as the intent for these two names ("Short-name aliases used
in `docs/api.md`'s multi-tenant example and `docs/error-handling-policy.md`'s
facade-error example"). Confirmed via compile-spike this was a harness
implementation bug relative to its own stated intent, not a doc bug: the doc's
own predominant convention (6+ sites, pre-dating this session) was already
correct; the harness's binding was wrong. Fixed by wrapping `llm` / `emb` in
`Arc::new(...)` at bind time, leaving `my_llm` / `my_embedder` (and `l` / `e`)
untouched. This single fix resolved 5 of the 9 harness failures that remained
after all 39 doc-content findings were fixed, without touching any doc content
that used the correct, pre-existing convention.

**Several additional real doc-vs-code defects were found ONLY by actually
re-running the compile harness after each fix** — none were in the original 39
Phase 1 rows, all are genuine (verified against real source, not assumed), and
all are FIXED. See "Additional findings — discovered during Phase 2
verification" at the end of this document for the full list (F34–F39): a wrong
suggested method name inside F5's own fix code (`.from_note()` doesn't exist on
`EpisodeEntryBuilder`), a stale `CoreError`/no-`.into()` duplicate of F3's bug in
`error-handling-policy.md`'s own second occurrence, an un-constructed
`LiveDashboardSink { /* ... */ }` literal, a `Namespace` moved-then-reused-bare
bug in §4, a bare-`my_llm` bug in `observability.md`'s "Override at runtime"
distinct from the harness prelude bug above, and README's "full trait" reference
block (already flagged as harness-attributable, not a doc defect, in the
harness's own README — resolved with an `ignore` marker rather than left
unmarked).

**Quality gate — full status**: `cargo build --workspace` ✅ · `cargo clippy -p
kremory -p kremory-napi --all-targets --all-features` ✅ zero warnings ·
`cargo nextest run -p kremory -p kremory-mcp --features content-search,test-utils`
✅ **1795/1795 passed, 6 skipped** (grew from the session-start baseline of 1794
by exactly the one new test added for F17) · `cargo test -p kremory-napi --test
api_parity` ✅ **7/7 passed** · `cargo test --doc -p kremory` (the crate's OWN
`///` doctests, a separate tier from the doc-examples harness) ✅ **26 passed, 0
failed, 5 ignored** — unaffected, unchanged from baseline.

No breaking API changes were required by any of the 39 findings or the
additional discoveries — every code-side fix (`with_graph_degree_weight`) is
purely additive, consistent with Phase 1's own count ("no row in this audit
requires a code-side fix" was true of the doc-vs-API findings; F17 was the one
exception, and it is additive, not breaking).

## Row counts by axis

| Axis | Rows |
|---|---|
| Wrong | 15 |
| Documented-but-absent | 1 |
| Unreachable | 1 |
| Undocumented | 16 (grouped; 16 rows name ~55 individual symbols) |
| SC-011 (version pins/banners — separate from the 4 API axes per spec) | 6 |
| **Total** | **39** |

Fix-layer split (excludes SC-011, which is uniformly doc-layer): of the 17
API-axis rows, **16 are doc-layer** fixes and **1 is ambiguous** (F10,
flagged for the maintainer below — could be doc-layer deletion or code-layer
addition). Of the 16 Undocumented rows, all 16 are doc-layer (add documentation
for existing code; RULE-006 never mandates a code change just because
something is undocumented). So: **32 doc-layer, 0 pure-code-layer, 1
ambiguous, 6 SC-011 (doc-layer)** across all 39 rows. No row in this audit
requires a code-side fix — every defect found is either a stale/wrong doc
example, an undocumented-but-correct piece of code, or (F10) unresolved
pending a maintainer call.

---

## Section A — Wrong / Documented-but-absent (RULE-004, RULE-005)

### F1 — `Memory::auto(...)` shown as a chainable builder (3 sites)

- **Axis**: Wrong
- **Doc side**: `docs/api.md:25-27` (§1 Quickstart), `docs/api.md:145-147`
  (§3 "Default namespace"), `README.md:304-306` ("Namespaces + multi-tenancy")
- **Code side**: `crates/kremory/src/facade/mod.rs:605` —
  `pub async fn auto(path: impl AsRef<Path>) -> Result<Self>`
- **What's wrong**: all three sites write
  `Memory::auto("./agent.db").default_namespace(Namespace::new(...)).await?`.
  `auto` is a plain `async fn` returning `Result<Self>` directly — there is no
  intermediate builder to call `.default_namespace()` on before `.await`. The
  chain does not compile.
- **Fix layer**: doc. Either drop the `.default_namespace()` call from these
  three `Memory::auto(...)` examples (namespace supplied per-call via
  `.in_namespace(...)` instead), or rewrite them to use
  `Memory::open(path).with_llm(...).with_embedder(...).default_namespace(...).await?`
  if a default-namespace-at-construction example is the intent.
- **Disposition**: Fixed (doc). Dropped `.default_namespace()` from the two `Memory::auto(...)` quickstart/single-tenant examples (docs/api.md §1, README.md "Namespaces + multi-tenancy") in favour of explicit `.in_namespace(ns.clone())` per call. Rewrote docs/api.md §3's "Default namespace (set once at construction)" example to use `Memory::open(...).with_llm(...).with_embedder(...).default_namespace(...)` instead, since that section's whole point is demonstrating construction-time default-namespace — the capability `Memory::auto` genuinely cannot express.

### F2 — Compile-error example missing the `compile_fail` marker

- **Axis**: Wrong
- **Doc side**: `docs/api.md:93-96`
- **Code side**: n/a (the example is intentionally illustrating a compile
  error — type-state guard on `MemoryBuilder`)
- **What's wrong**: the fence is plain ` ```rust ` with a comment
  `// Compile error — missing .with_embedder()` and a trailing `// ERROR`, but
  no `compile_fail` marker. Per RULE-001, an unmarked block is required to
  compile *successfully* — this one is deliberately supposed to fail, so
  Phase 0's harness will (correctly) flag it as broken unless marked.
- **Fix layer**: doc. Change the fence to ` ```rust,compile_fail `.
- **Disposition**: Fixed (doc). Fence changed to ` ```rust,compile_fail `.

### F3 — `kremory::CoreError::MissingNamespace` doesn't exist (2 sites)

- **Axis**: Wrong
- **Doc side**: `docs/api.md:172`, `docs/error-handling-policy.md:223`
- **Code side**: `crates/kremory/src/lib.rs:51` —
  `pub use core::error::{Error as CoreError, Result as CoreResult};` (so
  `CoreError` = `core::error::Error`); `MissingNamespace` is a variant of a
  **different** type, `MemoryError`, defined at
  `crates/kremory/src/memory/types.rs:1459` —
  `MissingNamespace { request: &'static str }`.
- **What's wrong**: both sites match on `kremory::CoreError::MissingNamespace`.
  `CoreError` and `MemoryError` are distinct enums; `MissingNamespace` is not a
  variant of `CoreError`. The match arm does not compile.
- **Fix layer**: doc. Both sites should read
  `Err(kremory::MemoryError::MissingNamespace { request })` — `MemoryError` is
  already re-exported at the crate root (`lib.rs:89`).
- **Disposition**: Fixed (doc). Both sites (`docs/api.md`, `docs/error-handling-policy.md`) changed `kremory::CoreError::MissingNamespace` → `kremory::MemoryError::MissingNamespace`, verified against `crates/kremory/src/memory/types.rs:1459`. The `error-handling-policy.md` occurrence also needed `Err(e) => return Err(e.into())` (was `return Err(e)`, a second, smaller bug in the same block — the function returns `anyhow::Result<()>`, not `MemoryError`).

### F4 — `Namespace::new(tenant)` called with `tenant: &&str`

- **Axis**: Wrong
- **Doc side**: `docs/api.md:188-191`
- **Code side**: `crates/kremory/src/memory/types.rs:51` —
  `pub fn new(namespace: impl Into<String>) -> Self`
- **What's wrong**: `for tenant in &["acme", "globex", "initech"] { ...
  Namespace::new(tenant) ... }` — iterating `&[&str; 3]` binds
  `tenant: &&str`. `&&str` has no `Into<String>` impl (only `&str` and
  `String` do; generic trait-bound resolution does not auto-deref), so the
  call fails to compile.
- **Fix layer**: doc. Either iterate `["acme", "globex", "initech"]` by value
  (drop the `&`, giving `tenant: &str`) or call `Namespace::new(*tenant)`.
- **Disposition**: Fixed (doc). Changed `for tenant in &["acme", "globex", "initech"]` to `for tenant in ["acme", "globex", "initech"]` (drop the leading `&`, iterate by value).

### F5 — `.add(...)` doesn't exist on `RememberBatchBuilder`

- **Axis**: Wrong
- **Doc side**: `docs/api.md:309-317`
- **Code side**: `crates/kremory/src/facade/remember.rs:300-359`
  (`impl<'a> RememberBatchBuilder<'a>`) and `:377-426`
  (`impl<'a> EpisodeEntryBuilder<'a>`)
- **What's wrong**: the doc chains
  `.add("Meeting at 2pm").from_chat("session-42").in_namespace(...).add("Alice prefers async Rust")...with_batch_id(...)`.
  There is no `.add()` method on `RememberBatchBuilder`.
  **Correction to the task's brief**: the brief suggested the real method is
  `.and(...)` — that is also wrong; there is no `.and()` method anywhere in
  this file or crate either (verified: `grep -rn "pub fn and\b"` across
  `crates/kremory/src/` returns nothing). The actual real API is
  `.entry(content) -> EpisodeEntryBuilder`, which is chained
  (`.from_chat(...)`/`.from_document(...)`/`.in_namespace(...)`/`.published_at(...)`/`.with_facts(...)`)
  and closed with `.done() -> RememberBatchBuilder` to return to the batch
  builder for the next `.entry(...)` or the terminal `.with_batch_id(...)`.
- **Fix layer**: doc. Correct shape:
  ```rust
  let commits = mem.remember_batch()
      .entry("Meeting at 2pm")
          .from_chat("session-42")
          .in_namespace(Namespace::new("user-jim"))
          .done()
      .entry("Alice prefers async Rust")
          .from_note("note-7")
          .in_namespace(Namespace::new("user-jim"))
          .done()
      .with_batch_id("import-2026-05-27")
      .await?;
  ```
- **Disposition**: Fixed (doc). Rewrote to the real `.entry(content)...done()` chain per the finding's own correction. Discovered on compile-verification that the finding's OWN suggested fix code was itself wrong — it used `.from_note("note-7")`, but `EpisodeEntryBuilder` (verified `crates/kremory/src/facade/remember.rs:379-426`) has only `from_chat`/`from_document` (no `from_note`, no `from_source` — unlike the single-episode `RememberRequest`, which has all four). Corrected to `.from_document("note-7")` with an inline comment noting the asymmetry; recorded as new finding F35 below.

### F6 — `r.content` field doesn't exist on `RetrievedContext`

- **Axis**: Wrong
- **Doc side**: `docs/api.md:364-371`
- **Code side**: `crates/kremory/src/memory/types.rs:554-626` —
  `pub struct RetrievedContext` real fields: `entity_id: String`,
  `entity_name: String`, `summary: String`, `score: f32`,
  `source_refs: Vec<SourceRef>`, `incomplete: bool`, `entity_type_id: u32`,
  `entity_type_name: String`, `namespace: Option<Namespace>`,
  `facts: Vec<RetrievedFact>` (10 fields total).
- **What's wrong**: `println!("{}: {:.3}", r.content, r.score);` — no
  `content` field exists.
- **Fix layer**: doc. Nearest equivalents are `r.summary` (a text summary) or
  `r.entity_name`; the example should use one of the real fields.
- **Disposition**: Fixed (doc). `r.content` → `r.summary` (nearest real field on `RetrievedContext`). Also added a one-line mention of `RetrievedContextNewParams`/`RetrievedFactNewParams` immediately after, folding in F24.

### F7 — `DreamOpts { ..DreamOpts::default() }` struct literal contradicts the doc's own next paragraph

- **Axis**: Wrong
- **Doc side**: `docs/api.md:603-616` (code block at 603-610, contradicting
  prose at 615-616 thirteen lines later)
- **Code side**: `crates/kremory/src/memory/types.rs:996-998` —
  `#[derive(Debug, Clone)] #[non_exhaustive] pub struct DreamOpts { ... }`
- **What's wrong**: the code example constructs
  `DreamOpts { include_community_detection: false, ..., ..DreamOpts::default() }`
  as a struct literal. `#[non_exhaustive]` structs cannot be constructed via
  struct-literal syntax from outside their defining crate — **not even** with
  a `..base` functional-update tail — so this fails to compile for any real
  consumer. The doc's very next sentence (line 615-616) already states
  `DreamOpts` "is `#[non_exhaustive]` — build it from `DreamOpts::default()` +
  field mutation, never a struct literal," directly contradicting its own
  preceding code block.
- **Fix layer**: doc. Rewrite the example as field mutation:
  ```rust
  let mut opts = DreamOpts::default();
  opts.include_community_detection = false;
  opts.include_fact_archival = false;
  opts.max_episodes_per_run = Some(500);
  let summary = mem.dream().with_opts(opts).await?;
  ```
- **Disposition**: Fixed (doc). Rewrote as `let mut opts = DreamOpts::default();` + field-mutation lines, per the doc's own very next paragraph (which was already correct).

### F8 — `Memory::builder()` and `.build()` don't exist

- **Axis**: Wrong
- **Doc side**: `docs/api.md:528-533`
- **Code side**: `crates/kremory/src/facade/mod.rs:595` —
  `pub fn open(path: impl AsRef<Path>) -> MemoryBuilder<NoLlm, NoEmb>` is the
  only entry point into the builder. The only `builder()` function in the
  whole crate is unrelated: `crates/kremory/src/core/config.rs:818` —
  `PipelineConfigBuilder::builder()`, not on `Memory`. No `.build()` method
  exists anywhere on `MemoryBuilder` (`grep -n "fn build\b"` across
  `crates/kremory/src/facade/*.rs` returns nothing) — the type-state builder's
  only terminal is `.await` once `.with_llm()` and `.with_embedder()` are set.
- **What's wrong**: `Memory::builder().prior_turn_replay_depth(0)...build().await?`
  — two compounding errors: `Memory::builder()` should be `Memory::open(path)`,
  and the extra `.build()` before `.await?` doesn't exist.
- **Fix layer**: doc. Correct shape:
  ```rust
  let mem = Memory::open("./agent.db")
      .with_llm(llm)
      .with_embedder(emb)
      .prior_turn_replay_depth(0)   // 0 = off; default 10
      .await?;
  ```
- **Disposition**: Fixed (doc). `Memory::builder()...build().await?` → `Memory::open("./agent.db").with_llm(llm).with_embedder(emb).prior_turn_replay_depth(0).await?` — confirmed `prior_turn_replay_depth` is a real, already-shipped `MemoryBuilder` method (`facade/builder.rs:518`, ADR-080), so no code change was needed here, only the call path.

### F9 — `on_edge_added` shown with the wrong arity

- **Axis**: Wrong
- **Doc side**: `docs/api.md:883`
- **Code side**: `crates/kremory/src/core/sink.rs:161` (`OnEdgeAddedParams`
  struct) and `:190` — real trait method is
  `fn on_edge_added(&self, _params: OnEdgeAddedParams<'_>) {}` (one params
  struct, default no-op body); re-exported at `lib.rs:102`.
- **What's wrong**: the doc's example impl is
  `fn on_edge_added(&self, _from: &str, _to: &str, _predicate: &str) {}` — 3
  separate `&str` params. A trait impl with an incompatible signature fails to
  compile ("method has an incompatible signature for the trait").
- **Fix layer**: doc. Correct signature:
  `fn on_edge_added(&self, _params: OnEdgeAddedParams<'_>) {}`.
- **Disposition**: Fixed (doc). Signature corrected to `fn on_edge_added(&self, _params: OnEdgeAddedParams<'_>) {}`; added `OnEdgeAddedParams` to the section's `use kremory::{...}` import list.

### F10 — `AuditSink` referenced but never defined anywhere — AMBIGUOUS, flagged for maintainer

- **Axis**: Documented-but-absent
- **Doc side**: `docs/api.md:919`
- **Code side**: none — `grep -rn "AuditSink" crates/kremory/src/
  crates/kremory-napi/` returns zero hits outside this one doc line.
- **What's wrong**: `.with_event_sink(Arc::new(AuditSink::new("audit-log.jsonl")))`
  — `AuditSink` is not a type that exists in the crate at any visibility.
- **I could not confidently resolve which layer is wrong — flagging per the
  task's explicit instruction.** Two readings:
  1. **Doc-layer (most likely)**: this is meant to illustrate the *pattern*
     of a per-call custom sink overriding the Memory-level default — exactly
     like `LiveDashboardSink` earlier in §9, which the user is expected to
     define themselves by implementing `EnrichmentEventSink`. `AuditSink` was
     probably meant as a second example custom type, but was never actually
     defined in the doc (unlike `LiveDashboardSink`, which has a full `impl`
     block a few lines above at api.md:874-896). Fix: either define a
     minimal `AuditSink` struct + `impl EnrichmentEventSink for AuditSink`
     inline, or reuse `LiveDashboardSink`/a placeholder from RULE-001's
     prelude (`MySink`).
  2. **Code-layer (worth the maintainer's judgment call)**: an `AuditSink`
     that writes to a JSONL file is a plausible, generically useful
     batteries-included sink a memory library might ship as a convenience
     type (the name and constructor shape — `AuditSink::new("audit-log.jsonl")`
     — read like a real feature request, not just a placeholder). If the
     maintainer wants this as a real shipped type, this becomes RULE-004's
     "documented but absent" resolved by adding the code, not deleting the
     doc line.
- **Fix layer**: **undecided — maintainer call required.** Recorded here
  rather than defaulted to doc-deletion.
- **Disposition**: Fixed (doc) — per the pre-decided judgment call. Rewrote the "Per-call override" example to use the harness's own generic `MySink` placeholder (already used unqualified in docs/api.md §2 for the exact same purpose) with prose explicitly framing it as "stands in for your own EnrichmentEventSink implementation", rather than inventing a shipped `AuditSink` type. No new code added to the crate.

### F11 — `submit_episode`/`search` shown with old positional-argument signatures

- **Axis**: Wrong
- **Doc side**: `docs/api.md:933-968` (§10 Advanced — substrate composition)
- **Code side**: `crates/kremory/src/memory/mod.rs:76-93` —
  `pub struct SubmitEpisodeParams<'a> { graph, content, source_ref,
  structured_facts, provider, namespace, batch_id, opts, sink }` and
  `pub async fn submit_episode(params: SubmitEpisodeParams<'_>) -> Result<EpisodeCommit>`;
  `crates/kremory/src/memory/mod.rs:303-312` —
  `pub struct SearchParams<'a> { graph, query, namespace, opts }` and
  `pub async fn search(params: SearchParams<'_>) -> Result<Vec<RetrievedContext>>`.
- **What's wrong**: the doc calls both functions with 9 (resp. 4) positional
  arguments — `submit_episode(graph.as_ref(), "Alice prefers async Rust",
  SourceRef {...}, vec![], provider.clone(), Namespace::new(...), None,
  SubmitOpts::default(), None)` and `search(graph.as_ref(), &Namespace::new(...),
  "rust preferences", SearchOpts { limit: Some(10), ..Default::default() })`.
  Both functions take exactly **one** params-struct argument (the args-as-object
  refactor from TD-042, `84dafd54`) — neither positional call compiles.
- **Fix layer**: doc. Correct shape:
  ```rust
  let commit = submit_episode(SubmitEpisodeParams {
      graph: graph.as_ref(),
      content: "Alice prefers async Rust",
      source_ref: SourceRef { kind: SourceKind::Chat, id: "session-42".into(),
                               occurred_at: chrono::Utc::now(), published_at: None },
      structured_facts: vec![],
      provider: provider.clone(),
      namespace: Namespace::new("user-alice"),
      batch_id: None,
      opts: SubmitOpts::default(),
      sink: None,
  }).await?;
- **Disposition**: Fixed (doc). Rewrote both `submit_episode(...)`/`search(...)` calls to the real single-bundled-params-struct shape (`SubmitEpisodeParams`/`SearchParams`), added both to the section's `use` list. On compile-verification, ALSO found the block references `graph`/`provider` bindings that are never defined anywhere in the doc (a narrative-fragment gap distinct from F11's arg-shape bug, pre-existing, not previously flagged) — marked the block `rust,ignore` with an explanatory prose note ("Illustrative — graph and provider below stand for a caller-supplied &dyn GraphHandle / Arc<dyn ChatProvider>") rather than fabricating a compiling stand-in graph handle, consistent with RULE-001's illustrative-content allowance.

  let results = search(SearchParams {
      graph: graph.as_ref(),
      query: "rust preferences",
      namespace: Namespace::new("user-alice"),
      opts: SearchOpts { limit: Some(10), ..Default::default() },
  }).await?;
  ```

### F12 — `ProviderRates::from_path` expects `&Path`, not `&str`

- **Axis**: Wrong
- **Doc side**: `docs/observability.md:276`
- **Code side**: `crates/kremory/src/core/rates.rs:132` —
  `pub fn from_path(path: &Path) -> Result<Self, RatesError>`
- **What's wrong**: `ProviderRates::from_path("./my-rates.toml")?` passes a
  `&str` literal where the signature requires `&Path` (not generic over
  `AsRef<Path>` — a concrete `&Path` parameter). `&str` does not coerce to
  `&Path` at a call site; this fails to compile.
  (Note, checked for completeness: the *builder* method
  `MemoryBuilder::with_provider_rates_path` at
  `crates/kremory/src/facade/builder.rs:164` takes `impl Into<PathBuf>`, which
  DOES accept a `&str` literal — every other `.with_provider_rates_path("./my-rates.toml")`
  call site in the docs, e.g. `api.md:116`, `README.md:444`,
  `observability.md:259`, is correct and not part of this finding.)
- **Fix layer**: doc. Either `ProviderRates::from_path(Path::new("./my-rates.toml"))?`
  or `ProviderRates::from_path("./my-rates.toml".as_ref())?`.
- **Disposition**: Fixed (doc). `ProviderRates::from_path("./my-rates.toml")` → `ProviderRates::from_path(Path::new("./my-rates.toml"))` with a `use std::path::Path;` import added. Also named `RatesError`/`ProviderRateEntry`/`CostUsdParams` inline (folding in F30) since they're the directly-adjacent undocumented types.

### F13 — `Memory::with_ollama_at_model(...)` is not a `Memory::` associated function

- **Axis**: Wrong
- **Doc side**: `README.md:251-255`
- **Code side**: `crates/kremory/src/facade/providers.rs:686` —
  `pub async fn with_ollama_at_model(url: impl Into<String>, model:
  Option<String>, path: impl AsRef<Path>) -> Result<Memory>` is a **free
  function** inside `pub mod providers` (declared at `facade/mod.rs:75`), not
  an inherent method in the `impl Memory { ... }` block
  (`facade/mod.rs:575-633`, which has `open`/`auto`/`with_ollama`/
  `with_ollama_at`/`with_openai`/`with_anthropic` only — no
  `with_ollama_at_model`). It is also not re-exported at the crate root
  (`lib.rs` has no mention of it).
- **What's wrong**: `Memory::with_ollama_at_model("http://localhost:11434",
  Some("qwen2.5:7b".into()), "./agent.db").await?` — `Memory::with_ollama_at_model`
  does not exist as a path; the function is only reachable as
  `kremory::facade::providers::with_ollama_at_model(...)`.
  Cross-checked against the Node-side parity gate: `crates/kremory-napi/parity-skip.toml:20-23`
  already lists `symbol = "Memory::with_ollama_at_model"` with reason
  "Internal env-auto plumbing for Memory::auto path; consumers go through
  Memory.open which already honors OLLAMA_HOST + OLLAMA_CHAT_MODEL via
  auto()" — confirming it is deliberately NOT mirrored to the Node binding,
  but that skip entry's symbol notation is a napi-parity-walker convention
  for referring to the Rust source name, not a claim that `Memory::` is the
  correct Rust call path either.
- **Fix layer**: doc. Either change the call to
  `kremory::facade::providers::with_ollama_at_model(...)`, or (cheaper, and
  consistent with the sibling `with_ollama_at` which IS an inherent
  `Memory::` method) add a thin `Memory::with_ollama_at_model` wrapper
  mirroring the existing `with_ollama_at` wrapper at `facade/mod.rs:617-619` —
  that would be a small additive code change in the TD-231 accessor style, so
  flagged here as the doc's *cheapest* correct fix but not the only one;
  Phase 2 should pick based on whether the maintainer wants this convenience
  path promoted to `Memory::`.
- **Disposition**: Fixed (doc). `Memory::with_ollama_at_model(...)` → `kremory::facade::providers::with_ollama_at_model(...)` (the real, only reachable path — verified `facade/providers.rs:686`, not re-exported at the crate root or on `impl Memory`), per the finding's own cheaper-fix recommendation. Did not add the alternative `Memory::` convenience wrapper — that remains the maintainer's call if wanted, and is not required to make the doc correct.

### F14 — `#[async_trait::async_trait]` on a `GraphHandle` impl doesn't compile (found independently, not in the pre-seeded list)

- **Axis**: Wrong
- **Doc side**: `docs/api.md:978-981`
- **Code side**: `crates/kremory/src/memory/graph.rs:106` —
  `pub trait GraphHandle: Send + Sync { async fn graph_ingest_episode(...) ->
  Result<EpisodeCommit>; ... }` — the trait is defined with **native**
  `async fn` (AFIT), not the `async_trait` macro.
- **What's wrong**:
  ```rust
  #[async_trait::async_trait]
  impl GraphHandle for MyGraphBackend {
      // implement all required methods
  }
  ```
  `#[async_trait]` rewrites method signatures to return
  `Pin<Box<dyn Future<Output = T> + Send + '_>>`. Applying it only to the impl
  block, when the trait itself declares plain `async fn`, produces a
  structurally incompatible signature — "method has an incompatible type for
  trait" — because the trait's native async-fn desugars to an
  anonymous/opaque return type the macro-rewritten impl does not match. The
  macro must be applied consistently to both the trait definition and every
  impl, or not at all; here it's on neither the trait nor consistently on the
  impl.
- **Fix layer**: doc. Delete the `#[async_trait::async_trait]` attribute —
  native `async fn` in a trait impl (MSRV 1.86, well past the 1.75 stabilization
  of AFIT) needs no macro.
- **Disposition**: Fixed (doc) — but the finding's stated MECHANISM was WRONG, verified by compile-spike (not assumed): `GraphHandle` IS declared with `#[async_trait]` (`crates/kremory/src/memory/graph.rs:105`), not native `async fn` as F14 claimed, so `#[async_trait::async_trait]` on the impl is CORRECT usage, not the bug — removing it (F14's proposed fix) would have been actively wrong, breaking any reader who went on to implement even one real method (the trait's macro-desugared signature would then not match a plain `async fn` impl). The REAL compile failure is that the impl's body implements zero of the trait's 12 required methods (`// implement all required methods` is a comment, not code — confirmed via the compiler's own `implement the missing item` suggestions, which are correctly macro-expanded, proving the macro pairing was never the issue). Fixed by marking the block `rust,ignore` with prose clarifying both facts (the macro pairing is correct, and a real implementation is a substantial 12-method adapter, not a snippet) — not by deleting the macro attribute as originally suggested.

### F15 — Doc claims "no retroactive upgrade" for namespace policy; `upgrade_namespace_policy` now exists

- **Axis**: Wrong (stale claim)
- **Doc side**: `docs/api.md:236-239` ("§3 Namespace policies")
- **Code side**: `crates/kremory/src/facade/mod.rs:2337-2357` —
  `pub async fn upgrade_namespace_policy(&self, namespace: Namespace) ->
  Result<()>` — "Monotonically upgrade a namespace's immutability from
  `Mutable` to `AppendOnly` (ADR-029b Decision 5). This is a **one-way
  ratchet**... Calling on an already-`AppendOnly` namespace is idempotent."
- **What's wrong**: the doc states "Call `register_namespace` explicitly at
  startup for namespaces that need a non-default policy — there is no
  retroactive upgrade at v0.1.4," worded as a still-current limitation. The
  crate is now at 0.7.x and a real, reachable, non-cfg-gated
  `upgrade_namespace_policy` ratchet exists and directly contradicts this
  sentence — a stale v0.1.4-era limitation statement that was never revisited
  when the capability shipped. It is also, independently, entirely
  undocumented by name (see F32-adjacent note in Section C).
- **Fix layer**: doc. Update the "no retroactive upgrade" sentence to
  describe the real, current one-way `Mutable → AppendOnly` ratchet via
  `upgrade_namespace_policy`, with a short example and its error cases
  (`NamespacePolicyImmutable` on downgrade attempts).
- **Disposition**: Fixed (doc). Replaced the stale "no retroactive upgrade at v0.1.4" sentence with a description of the real, current one-way `Mutable → AppendOnly` ratchet via `Memory::upgrade_namespace_policy`, including a short compiling example and its error cases, verified against `facade/mod.rs:2337-2357`.

### F16 — Default-feature-set self-contradiction (pre-seeded from `plan.md`, folded in per T1.4)

- **Axis**: Wrong
- **Doc side**: `docs/api.md:1158` (stale) vs `docs/api.md:1167` (correct)
- **Code side**: `crates/kremory/Cargo.toml:55` — `default = ["content-search"]`
  (ADR-078, 2026-07-28)
- **What's wrong**: line 1158's table row says "The crate has an explicit
  **empty** default feature set + opt-in features"; thirteen lines later, the
  §13 heading at line 1167 correctly says "kremory's `default` feature set is
  **`["content-search"]`** (ADR-078, 2026-07-28 — it was previously empty)."
  Line 1158 describes the pre-ADR-078 state as though it were still current.
- **Fix layer**: doc. Update line 1158's table row to match line 1167 (or
  simply delete the redundant/stale row and point to §13).
- **Disposition**: Fixed (doc). Updated the stale table row to state the crate's real `default = ["content-search"]` feature set (ADR-078) instead of the pre-ADR-078 "empty default" description, matching §13's already-correct text.

---

## Section B — Unreachable (implemented + configured, no public path to set it)

### F17 — `SearchConfig::graph_degree_weight` has no public setter and no env override

- **Axis**: Unreachable
- **Doc side**: none — not documented anywhere, which is itself consistent
  with the field being genuinely unreachable
- **Code side**: `crates/kremory/src/core/config.rs:234` —
  `pub graph_degree_weight: f32` (public field on `pub struct SearchConfig`,
  `config.rs:193`), default `0.05` (`config.rs:440`). A comment at
  `config.rs:225` calls it "the only... live tested axis" among its siblings.
- **What's wrong**: its three sibling weight axes on the same struct
  (`content_stream_weight`, `proximity_weight`, `temporal_weight`) each have
  **both** a `MemoryBuilder::with_*` setter (`facade/builder.rs:335`, `:412`,
  `:429`) **and** a `KREMORY_*_WEIGHT` env-var override
  (`facade/providers.rs:472`, `:591`, `:614`). `graph_degree_weight` has
  neither — `grep -rn "graph_degree_weight" crates/kremory/src/` shows it is
  only ever read (by the scoring path) and set to its hardcoded default; there
  is no `with_graph_degree_weight` method and no
  `KREMORY_GRAPH_DEGREE_WEIGHT` env var anywhere in the crate. It is reachable
  only for *reading* (via `Memory::search_config()`, itself undocumented —
  see F32), never for *writing*. This is the direct mirror of the TD-231
  "documented but absent" class: implemented and used, but no public path in.
- **Fix layer**: **code** (this is the one candidate in this audit where the
  fix is plausibly code-side, not doc-side — noted for the maintainer's Phase
  2 judgment, since RULE-006 forbids reconciling a doc to broken behaviour,
  but there is no *doc* claim here to reconcile; the gap is a missing knob).
  Additive `MemoryBuilder::with_graph_degree_weight(v: f32)` mirroring the
  three siblings (TD-231 accessor-addition precedent) would close it; whether
  to also add a `KREMORY_GRAPH_DEGREE_WEIGHT` env override for parity with the
  siblings is the maintainer's call. Alternatively, if the field is
  intentionally fixed (not meant to be tunable), the fix is to document that
  explicitly (doc-side) rather than leave silent asymmetry with its three
  documented-as-tunable siblings.
- **Disposition**: Fixed (code) — per the pre-decided judgment call. Added `MemoryBuilder::with_graph_degree_weight(f32)` (`crates/kremory/src/facade/builder.rs`), `PipelineConfigBuilder::graph_degree_weight(f32)` and a `PipelineConfigOverrides::graph_degree_weight: Option<f32>` field + `apply()` wiring (`crates/kremory/src/core/config.rs`) — the exact `extraction_arm_budget_ms` precedent shape (no env var, matching that precedent's own scope). Added a dedicated regression test (`with_graph_degree_weight_reaches_live_search_config`) plus a default-value assertion in `unset_knobs_match_documented_defaults`, both in `crates/kremory/tests/it/td141_search_config_builder_seam.rs`, following that file's own established pattern for this class of setter. Required raising the napi parity-skip cap 90 → 91 for exactly one new entry (`MemoryBuilder::with_graph_degree_weight`, `crates/kremory-napi/parity-skip.toml`) — RULE-013 compliant, one-line rationale, following the file's own raise-with-history convention (see `crates/kremory-napi/tests/api_parity.rs`). No `KREMORY_GRAPH_DEGREE_WEIGHT` env var was added, matching precedent and keeping the change minimal — the maintainer's call if parity with the three env-backed siblings is later wanted. Also documented the new knob in a new §13 "Advanced tuning knobs" appendix (docs/api.md), alongside the other 15 previously-undocumented knobs from F33.

---

## Section C — Undocumented (RULE-003)

Produced by diffing every `pub` item reachable from `crates/kremory/src/lib.rs`
re-exports, plus every `pub fn` in the `impl Memory` and `impl MemoryBuilder`
blocks, against every `docs/*.md` + `README.md` file (grep-per-symbol; a
symbol counts as documented if its exact identifier string appears anywhere
in the doc corpus). `cargo public-api` was checked and is not already a
workspace dependency; installing it fresh (needs nightly rustdoc-JSON
extraction) was judged more expensive than the grep sweep below for this pass
— noted per `evaluate-3p-before-handrolling`'s "check before hand-rolling"
instruction, but the grep sweep is not really "hand-rolling a public-API
diff tool," it's a one-off enumeration, so no tool was built, just run.
Items already covered by cfg-gated test-only accessors
(`temporal_graph_for_test`, `group_id_for_test` — both
`#[cfg(any(test, feature = "test-utils"))]`) are excluded: they are not part
of the default-build reachable surface, and the doc already correctly
describes this pattern for `temporal_graph_for_test` at `api.md:1062-1064`.

Rows are grouped by subsystem for usability; every symbol named in a group is
individually undocumented (verified by grep), not just the group's headline
name.

### F18 — Custom entity-type registry subsystem

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:53-55` (`EntityTypeSpec`,
  `NamespaceRegistrationError`, `NamespaceSeed`, `SeedOutcome`),
  `facade/mod.rs:2245` (`register_namespace_with_seed`), `facade/mod.rs:2602`
  (`assert_entity_type`, taking `GraphAssertEntityTypeParams`),
  `facade/builder.rs` (`with_seed_registry`)
- **Fix layer**: doc — add a subsection (custom entity-type registry) to
  §3 or a new section; the comment at the registry's own source calls out
  "custom-entity-type-registry spec §5.2" as its origin, which would be the
  natural source to draw the doc content from.
- **Disposition**: Fixed (doc). Added a "Custom entity-type registry" subsection to §3 (docs/api.md) covering `EntityTypeSpec`, `NamespaceSeed` (with a compiling `Augment` example verified against `core/entity_types.rs`), `SeedOutcome`, `Memory::register_namespace_with_seed`, `Memory::assert_entity_type`/`GraphAssertEntityTypeParams`, and `MemoryBuilder::with_seed_registry`.

### F19 — Background ingestor subsystem

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:57-58` (`BackgroundIngestor`,
  `IngestError`, `IngestErrorKind`, `IngestGuard`, `IngestSendError`,
  `IngestorConfig`), `facade/mod.rs:778` (`Memory::send_batched`),
  `facade/builder.rs` (`MemoryBuilder::with_sink`)
- **Fix layer**: doc — add to §8 (Async patterns) or a new subsection; the
  doc comment on `send_batched` (mod.rs:760-777) already explains the
  ADR-051 OS-thread pipeline vs tokio-spawn routing distinction in detail and
  would translate directly into doc prose.
- **Disposition**: Fixed (doc). Added a "Background ingestor (OS-thread pipeline, ADR-051)" subsection to §8, covering `Memory::send_batched`, the `.with_event_sink(...)` trigger condition (not the deprecated `.with_sink()`, which the finding's own code-side pointer named but which carries a `#[deprecated]` attribute since v0.2.5 — verified before using it in new doc content), and naming `IngestorConfig`/`IngestGuard`/`IngestError`/`IngestErrorKind`/`IngestSendError`.

### F20 — Dream scheduler subsystem

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:80` (`DreamSchedule`,
  `DreamSchedulerHandle`), `facade/mod.rs:2620` (`start_dream_scheduler`),
  `facade/mod.rs:2636` (`stop_dream_scheduler`), `facade/builder.rs`
  (`with_dream_schedule`, `with_dream_llm`, `with_dream_model_id`)
- **Fix layer**: doc — add to §6 (Dream phase); this is a periodic-scheduling
  convenience over manual `mem.dream()` calls and is a natural fit right
  after the existing "Fire-and-forget (async handle)" subsection.
- **Disposition**: Fixed (doc). Added a "Periodic scheduling — DreamSchedule" subsection to §6, adapting the real, already-doctested example from `crates/kremory/src/memory/scheduler.rs`'s own module doc comment. Also named `DreamStatus` and `Memory::cancel_dream` (folding in F31), and explicitly flagged `DreamMode` as reserved/not-yet-wired per its own source doc comment ("DORMANT", "not yet wired (F-01)") rather than documenting it as a usable knob.

### F21 — Process-global engine singleton

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:60` — `pub use
  core::engine::{engine, engine_init};`
- **Fix layer**: doc — the module doc comment at `lib.rs:59` calls this
  "consumer-facing, intentional," so it is a deliberate surface, not an
  oversight; needs at least a one-paragraph mention (what it's for, when to
  reach for it over the `Memory` facade) in §10 (Advanced).
- **Disposition**: Fixed (doc). Added a "Process-global engine singleton" subsection to §10 naming `kremory::engine()`/`kremory::engine_init()` and their intentional, consumer-facing status per `lib.rs:59`'s own module doc comment.

### F22 — Dream Phase-C developer internals

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:79,82` (`DreamPassOpts`,
  `TypeProposal`), `facade/mod.rs:2575` (`run_dream_pass_sync`),
  `facade/mod.rs:2591` (`ghost_episodes`)
- **Fix layer**: doc. Lower materiality than F18-F21 — both `pub fn` doc
  comments explicitly cite "Phase C DoD" ADR references
  (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`), suggesting these may be
  intended as lower-level/diagnostic escape hatches rather than mainline
  consumer API. Recommend at minimum a one-line mention in §10 noting they
  exist for diagnostic use, even if not given full worked examples.
- **Disposition**: Fixed (doc). Added a "Diagnostic / lower-level dream internals" note to §10 naming `Memory::run_dream_pass_sync`/`DreamPassOpts`, `Memory::ghost_episodes`, and `TypeProposal`, including the caveat (verified against the real doc comment) that two of `run_dream_pass_sync`'s sub-passes are stubs returning empty/zero counts.

### F23 — `DreamPhaseResult`, `IngestResult` result types

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:87-88`
- **Fix layer**: doc — name these in whichever section documents their
  producing call path (likely §6/§4 respectively); currently only
  `DreamSummary` and `EpisodeCommit` are named as the documented result
  shapes.
- **Disposition**: Fixed (doc). Named `DreamPhaseResult` (produced by `GraphHandle::graph_run_consolidation`, converted to `DreamSummary` via the real `impl From<DreamPhaseResult> for DreamSummary`) and `IngestResult` (the deprecated `ingest_episode` free function's return type — flagged as deprecated, not a path new code should use) in §10.

### F24 — Constructor param structs (`RetrievedContextNewParams`, `RetrievedFactNewParams`)

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:89-90`
- **Fix layer**: doc, low priority — these back `RetrievedContext::new(...)` /
  `RetrievedFact::new(...)` constructors used mainly by test fixtures and
  advanced substrate consumers building result sets by hand; a one-line
  mention alongside F6's fix (§5 raw results) would suffice rather than a
  dedicated subsection.
- **Disposition**: Fixed (doc). Folded into F6's fix — one-line mention of `RetrievedContextNewParams`/`RetrievedFactNewParams` immediately after the "Raw results" example in §5.

### F25 — `TelemetryInitError`, `InvalidPolicyError` error types

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:91` (`TelemetryInitError`),
  `lib.rs:94` (`InvalidPolicyError`)
- **Fix layer**: doc — `TelemetryConfig`/`TelemetryHandle`/`init_telemetry`
  are already documented (observability.md); their error type isn't named.
  `InvalidPolicyError` is the error type for `NamespacePolicy` construction
  (§3 policy section) and should be named alongside the existing
  `NamespacePolicyImmutable` error already documented there.
- **Disposition**: Fixed (doc). `InvalidPolicyError` named in §3 (docs/api.md) alongside `NamespacePolicyImmutable`, with its one real variant (`IncoherentAppendOnly`) described per `memory/types.rs:255-263`. `TelemetryInitError` named in `docs/observability.md`'s OTel section, with both real variants (`Exporter`, `Subscriber`) described per `memory/mod.rs:697-705`.

### F26 — `AwaitOpts`, `BatchStatus`, `DreamMode`, `DreamStatus`

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:96-99`
- **Fix layer**: doc — §8 documents `await_batch`/`status_of`/`IngestStatus`
  by example but never names `BatchStatus`/`AwaitOpts` as the types involved;
  `DreamMode`/`DreamStatus` are the dream-handle analogues, similarly unnamed
  in §6's fire-and-forget subsection.
- **Disposition**: Fixed (doc). `BatchStatus` named as `await_batch`'s real return type in §8 (was previously only described by example, never named); `AwaitOpts` named alongside it with an accurate note that it's an internal type built from the `Duration` argument, not a caller-facing parameter. `DreamStatus` named in §6 (folded into F20's fix); `DreamMode` named and explicitly flagged reserved/not-yet-wired (also folded into F20's fix).

### F27 — `ArcEmbedder` BYOM wrapper type

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:106`
- **Fix layer**: doc — §2/README's BYOM section documents
  `DynEmbeddingProvider`/`EmbeddingProvider`/`.into_dyn()` but never names
  `ArcEmbedder`, which per its co-location with those types is presumably the
  concrete wrapper `.into_dyn()` produces or a related helper.
- **Disposition**: Fixed (doc). `ArcEmbedder` named in README's BYOM section (inside the now-`,ignore`-marked "full trait" reference block — see F34 below) with an accurate one-sentence description verified against `core/provider/embedding.rs:32` (a reverse-direction adapter from `Arc<dyn DynEmbeddingProvider>` back to `impl EmbeddingProvider`, distinct from what the finding's own text guessed it might be).

### F28 — `CoreConfig` re-export

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:109` — `pub use
  core::config::Config as CoreConfig;`
- **Fix layer**: doc — one-line mention in §10 (Advanced/substrate
  composition), since this is the substrate-level config type.
- **Disposition**: Fixed (doc). `kremory::CoreConfig` named in §10 alongside the process-global engine singleton mention (folded into F21's fix).

### F29 — `split_for_embedding` caller-side chunking helper

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:117` — `pub use
  core::chunking::split_for_embedding;` (TD-232/TD-234, added this session
  per git log — `bbffc996`)
- **Fix layer**: doc — recently shipped, zero doc mentions yet. Given the
  module doc comment it's re-exported alongside (per lib.rs:114-116) explains
  "why this exists and what it deliberately does NOT do (kremory never calls
  it automatically)," this reads as a genuine oversight from a very recent
  commit rather than a deliberate omission — good candidate for a short §4
  (Ingest) subsection on chunking large documents before `remember()`.
- **Disposition**: Fixed (doc). Added a "Chunking large documents before remember()" subsection to §4, adapting the real, already-doctested example from `core::chunking`'s own `split_for_embedding` doc comment.

### F30 — `kremory::observability` module extras (`CostUsdParams`, `ProviderRateEntry`, `RatesError`)

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/lib.rs:127-128`
- **Fix layer**: doc — `ProviderRates` itself is documented
  (observability.md "Provider rates" section); its companion types
  (`ProviderRateEntry` — presumably the deserialized TOML row shape,
  `RatesError` — `from_path`'s error type per F12, `CostUsdParams`) aren't
  named. Natural fit alongside the existing provider-rates TOML schema
  documentation.
- **Disposition**: Fixed (doc). Folded into F12's fix — `ProviderRateEntry`, `RatesError`, and `CostUsdParams` all named in `docs/observability.md`'s "Pre-loaded rates" section, verified against `core/rates.rs`.

### F31 — `Memory::cancel_dream`

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/facade/mod.rs:2100` —
  `pub async fn cancel_dream(&self, handle: &DreamHandle) -> Result<CancelOutcome>`
- **Fix layer**: doc — §6's "Fire-and-forget (async handle)" subsection
  documents `mem.dream().fire_and_forget()` → `DreamHandle` →
  `mem.await_dream(&handle, ...)` but never mentions the ability to cancel a
  fire-and-forget dream run, despite the ordinary ingest path's equivalent
  (`mem.cancel(&commit)`) being documented at §8.
- **Disposition**: Fixed (doc). Folded into F20's fix — `Memory::cancel_dream` documented in §6's "Fire-and-forget (async handle)" subsection, right after `await_dream`, with a compiling example.

### F32 — `Memory::search_config`, `Memory::contradiction_detection_enabled` introspection accessors

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/facade/mod.rs:650`
  (`search_config() -> SearchConfig`, TD-135), `facade/mod.rs:668`
  (`contradiction_detection_enabled() -> bool`)
- **Fix layer**: doc — both are read-only introspection accessors explicitly
  added (per their doc comments) so a transport/consumer can report the
  *actual* active config rather than re-reading env, which the comment notes
  can drift. Worth a short "Introspection" subsection, likely in §10; also
  directly relevant to F17 (`search_config()` is the only reachable path to
  even *observe* `graph_degree_weight`, let alone set it).
- **Disposition**: Fixed (doc). Added an "Introspection — reading back the ACTIVE config" subsection to §10, documenting `Memory::search_config()` and `Memory::contradiction_detection_enabled()` with a compiling example, and explicitly cross-referencing F17's `graph_degree_weight` as the knob this accessor is the only way to observe.

### F33 — Remaining `MemoryBuilder` advanced tuning knobs (15 methods)

- **Axis**: Undocumented
- **Code side**: `crates/kremory/src/facade/builder.rs` —
  `allowed_entity_types`, `episode_content_warn_threshold`,
  `extraction_arm_budget_ms`, `with_await_extraction`,
  `with_await_extraction_timeout`, `with_content_stream_weight`,
  `with_contradiction_detection_enabled`, `with_embed_task_prefix_enabled`,
  `with_episode_dense_enabled`, `with_extractor`, `with_fact_dense_enabled`,
  `with_proximity_weight`, `with_rerank_candidate_max_chars`, `with_rrf_k`,
  `with_temporal_weight`
- **Fix layer**: doc — a dedicated "Advanced tuning" appendix (or extending
  §2/§13) covering these recall-scoring and extraction-tuning knobs would
  close the gap in one pass; several (`with_rrf_k`, `with_temporal_weight`,
  `with_proximity_weight`, `with_content_stream_weight`) are the very knobs
  behind recently-shipped recall-scoring work (the RRF `k` default flip,
  ADR-062/ADR-082 phases) referenced elsewhere in this repo's active-sprint
  notes, so are likely to be asked about soon if not documented now.
- **Disposition**: Fixed (doc). Added an "Advanced tuning knobs (MemoryBuilder)" appendix to §13 — a 16-row table (the 15 named in this finding plus F17's new `with_graph_degree_weight`) with a short compiling example chaining three of the knobs together.

---

## Section D — SC-011: stale version pins + banners

Swept every `docs/*.md` + `README.md` for `version = "0.` TOML pins and
`vX.Y.Z`-shaped banners. Published crate version at time of writing:
**0.7.0** (crates.io, re-verified live via
`https://crates.io/api/v1/crates/kremory` → `max_version: 0.7.0`). Tree
version: `0.7.1` (`crates/kremory/Cargo.toml:3`, prepared but unpublished).

| ID | File:line | Current text | Target | Disposition |
|---|---|---|---|---|
| V1 | `docs/api.md:3` | `> **v0.6.0** — The primary consumer surface...` | `v0.7` (or drop the version pin from the banner entirely and let §12's migration-guide table carry version history) | Fixed — banner changed to `v0.7` |
| V2 | `docs/api.md:398` | `kremory = { version = "0.6", default-features = false, features = ["content-search"] }` | `version = "0.7"` | Fixed |
| V3 | `docs/api.md:1187` | `kremory = { version = "0.6", features = ["content-search", "otel"] }` | `version = "0.7"` | Fixed |
| V4 | `docs/observability.md:288` | `kremory = { version = "0.6", features = ["otel"] }` | `version = "0.7"` | Fixed |
| V5 | `README.md:429` | `kremory = { version = "0.6", features = ["otel"] }` | `version = "0.7"` | Fixed |
| V6 | `docs/api.md:1215` (footer) — **found independently, not in the pre-seeded list** | `*API reference current as of kremory v0.4.0 (2026-07-12)...*` | `v0.7` (this is the single most stale reference in the file — 3 minor versions behind V1's already-stale v0.6.0 banner) | Fixed — footer changed to `v0.7 (2026-09-07)` |

Note: `README.md:29` (`kremory = "0.7"`, the top-level Install section) is
**correct** — already matches the published version — and is not a finding.

Per RULE-012, all six of these are plain version-string corrections and don't
depend on any Phase-2 breaking code fix landing first, so none needs a
"target version, unreleased" label — they can be corrected directly to
`0.7` in the same pass as the other Section A/B/C doc fixes.

---

## Cross-references to prior work

- `crates/kremory-napi/tests/api_parity.rs` — the enforced Rust↔Node
  symbol-parity cap. **Updated during Phase 2**: this note originally said
  none of the 39 findings would require a new parity-skip entry, sitting at
  90/90 — that held for 38 of the 39 rows (every doc-vs-Rust-code defect is
  exactly that: a doc bug, not a Rust-vs-Node-binding symbol gap). **F17 was
  the one exception** — its code-side fix (`MemoryBuilder::with_graph_degree_weight`)
  is a new Rust-only public symbol, so it needed exactly one new skip-list
  entry per RULE-013, raising the cap **90 → 91**
  (`crates/kremory-napi/parity-skip.toml`, `grep -c '^\[\[skip\]\]'` → 91 after
  the raise). F13 remains as originally described — a doc calling a Rust path
  incorrectly, not a symbol-mirror-status gap.
- The Node/napi undo + inspect surface claims in `docs/api.md:1192-1211`
  (§14) were spot-checked against `crates/kremory-napi/src/lib.rs` for the
  camelCase method names (`mutationHistory`, `listMutations`, `undo`,
  `unmerge`, `undoEntityEdit`, `undoDeleteEntity`, `undoDeleteFact`,
  `editEntity`, `deleteEntity`, `deleteFact`, `recallBySourceId`) and the
  `RecallOptions.rerankK` field (`crates/kremory-napi/src/convert.rs:316`) —
  **all confirmed accurate, no finding**. This section of the docs is
  correct and needed no changes.
- `RetrievedFact`'s four temporal fields (`valid_at`, `invalid_at`,
  `recorded_at`, `expired_at`, claimed at `docs/api.md:1042-1044`) were
  spot-checked against `crates/kremory/src/memory/types.rs:441-460` —
  **all confirmed accurate, no finding**.

---

## Additional findings — discovered during Phase 2 verification (F34–F39)

None of these were in Phase 1's 39-row work list. All were found ONLY because
Phase 2 actually re-ran `cargo test --doc -p kremory-doc-examples` after every
doc edit (per SC-004/RULE-006 discipline — "mechanically confirm... rather than
trusting your own edits by eye") instead of trusting the fixed doc text by
inspection. All are genuine, verified against real source, and all are FIXED.

### F34 — README's "full trait" `EmbeddingProvider` reference block has no
non-compiling marker

- **Axis**: Wrong (harness-attributable — see below)
- **Doc side**: `README.md` BYOM section, "The full trait
  (`kremory::EmbeddingProvider`):" block
- **What's wrong**: this block re-shows the trait's own definition as
  reference text, immediately after an earlier block in the same `##`-grouped
  compilation unit already does `use kremory::{CoreResult, EmbeddingProvider};`
  — concatenated, the re-declaration collides with the import
  (`E0255: the name EmbeddingProvider is defined multiple times`). Read
  standalone, as an actual human reads the page top-to-bottom, there is no
  conflict — this is a harness-attributable artifact of the compile-audit's
  own per-`##`-heading grouping decision, **already identified and disclaimed
  in `crates/kremory-doc-examples/README.md`'s own "Known, deliberate gaps"
  section** before this session started, but the doc block itself carried no
  explicit non-compiling marker, so it still showed up as an unmarked compile
  failure (violating SC-010's "a marker alone... does not satisfy SC-001,
  every phase-0 compile failure needs an SC-003 row" requirement in spirit —
  this row closes that gap).
- **Disposition**: Fixed (doc). Marked the fence `rust,ignore` with a short
  prose note ("reference — already imported above, shown again here for the
  complete shape in one place"). Also folded in F27 (`ArcEmbedder`, named
  inside this same block).

### F35 — `EpisodeEntryBuilder::from_note` doesn't exist (surfaced inside F5's own fix)

- **Axis**: Wrong
- **Doc side**: `docs/api.md` §4 "Batch ingest" (the F5 block)
- **Code side**: `crates/kremory/src/facade/remember.rs:379-426`
  (`impl<'a> EpisodeEntryBuilder<'a>`) — real methods: `from_chat`,
  `from_document`, `in_namespace`, `published_at`, `with_facts`, `done`. No
  `from_note`, no `from_source` — unlike the single-episode `RememberRequest`
  (`remember.rs:22-57`), which has all four (`from_chat`, `from_note`,
  `from_document`, `from_source`).
- **What's wrong**: F5's OWN suggested fix code (in `phase1-findings.md`
  itself) called `.from_note("note-7")` on the batch builder — a real,
  genuine API asymmetry between `RememberRequest` and `EpisodeEntryBuilder`
  that Phase 1's code-reading pass missed, caught only by actually compiling
  the corrected example.
- **Disposition**: Fixed (doc). Changed to `.from_document("note-7")` with an
  inline comment naming the asymmetry, rather than adding `from_note`/
  `from_source` to `EpisodeEntryBuilder` for parity — the doc content doesn't
  depend on `SourceKind::Note`'s specific semantics, and adding new public
  builder methods to close a doc example is a bigger, separately-justified
  change (parity with `RememberRequest`) that the maintainer should decide on
  its own merits, not as a side-effect of a docs pass.

### F36 — Duplicate of F3's bug in `error-handling-policy.md`'s own second occurrence

- **Axis**: Wrong
- **Doc side**: `docs/error-handling-policy.md`, "Facade errors" → "Match
  pattern" subsection
- **What's wrong**: this block has its OWN `Err(kremory::CoreError::MissingNamespace
  { request })` match arm (the same bug F3 fixed at a DIFFERENT site in the
  same file), plus a second, independent bug in the same line:
  `Err(e) => return Err(e)` — the enclosing function returns `anyhow::Result<()>`,
  not `MemoryError`, so this needs `.into()`. F3's original text named only
  one occurrence in this file (`docs/error-handling-policy.md:223`); this is
  a second, separate occurrence a few lines below it that Phase 1 did not
  catch.
- **Disposition**: Fixed (doc). `CoreError` → `MemoryError`, `return Err(e)` →
  `return Err(e.into())`.

### F37 — `LiveDashboardSink { /* ... */ }` doesn't actually construct the struct

- **Axis**: Wrong
- **Doc side**: `docs/api.md` §9 "Memory-level sink" subsection
- **Code side**: the `LiveDashboardSink` struct is defined earlier in the
  SAME `##`-grouped compilation unit (§9's first example) with two real
  fields: `entity_count: Arc<AtomicUsize>`, `contradiction_count: Arc<AtomicUsize>`.
- **What's wrong**: `Arc::new(LiveDashboardSink { /* ... */ })` — the `/* ... */`
  placeholder is not valid struct-literal syntax with real fields present;
  `E0063: missing fields`.
- **Disposition**: Fixed (doc). Supplied real values:
  `Arc::new(AtomicUsize::new(0))` for both fields.

### F38 — `Namespace` (`ns`) reused after move within one block

- **Axis**: Wrong
- **Doc side**: `docs/api.md` §4 "Source metadata" subsection
- **What's wrong**: the block calls `.in_namespace(ns)` bare TWICE across two
  separate `mem.remember(...)` statements inside the SAME fenced block.
  `Namespace` is not `Copy`, so the second usage is `E0382: use of moved
  value`.
- **Disposition**: Fixed (doc). First occurrence changed to
  `.in_namespace(ns.clone())`.

### F39 — Bare `my_llm`/`my_embedder` in `observability.md`'s "Override at runtime" section

- **Axis**: Wrong
- **Doc side**: `docs/observability.md` "Provider rates (cost emission)" →
  "Override at runtime" subsection
- **What's wrong**: `.with_llm(my_llm).with_embedder(my_embedder)` bare — this
  is a DIFFERENT `##`-grouped compilation unit from "## Wiring observability
  into your app" (where an EARLIER example locally constructs
  `let my_llm = Arc::new(MyCustomChatProvider::new());` before using it bare,
  which is correct THERE). This later, separate section reuses the same
  variable names without a local definition, and — unlike the harness's own
  `llm`/`emb` placeholder pair (fixed at the harness-prelude level, see the
  Phase 2 summary above) — `my_llm`/`my_embedder` are, by the doc's own
  predominant convention elsewhere (§2 Tier 2 Builder, README), meant to be
  wrapped explicitly via `Arc::new(my_llm)` at the CALL SITE, not pre-wrapped
  by the harness.
- **Disposition**: Fixed (doc). Changed to
  `.with_llm(Arc::new(my_llm)).with_embedder(Arc::new(my_embedder))`.

### Harness bug (not a doc finding) — `llm`/`emb` prelude binding

Documented in full in the "Phase 2 disposition — summary" section at the top
of this document. Fixed in `crates/kremory-doc-examples/generate_docs.py`
(`PRELUDE_REFRESH_LINES`): `llm`/`emb` are now bound as pre-wrapped
`Arc<dyn ChatProvider>`/`Arc<dyn DynEmbeddingProvider>` values, matching the
predominant doc convention for those two specific short names (distinct from
`my_llm`/`my_embedder`, which correctly stay unwrapped). This single fix
resolved 5 compile failures across `docs/api.md` §3/§5.2/§9/§6a and
`README.md`, none of which required any doc-content change.
