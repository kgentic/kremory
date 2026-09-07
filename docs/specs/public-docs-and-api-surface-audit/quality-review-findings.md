# Quality Review Findings — Docs Narrative/Consistency + API Sensibility

**Scope:** This is a *quality* review, distinct from the earlier correctness audit
(`phase1-findings.md`). Correctness (do the examples compile, do the methods exist) is
already verified and DONE. This review asks two different questions:

1. Reading the public docs cold, as a genuine newcomer would — is the *documentation* good
   (clear, well-paced, consistent, honest about why-before-how)?
2. Reading the public `MemoryBuilder` API against the Rust API Guidelines
   (rust-lang.github.io/api-guidelines, fetched live 2026-09-07, not from training memory) —
   does the *API design* make sense?

Read-only review. No files were modified except this one.

**Reading order followed (as a real newcomer would encounter it):** `README.md` →
`docs/getting-started.md` → `docs/api.md` → `website/sidebars.ts` +
`website/docusaurus.config.ts` (site structure) → `website/docs/intro.md` (the site's actual
homepage, since `sidebars.ts` puts `intro` first and it doesn't exist at repo-root `docs/`) →
`docs/observability.md` + `docs/comparison.md` (tone spot-check).

---

## Part A — Docs quality, read cold as a newcomer

### A1. [HIGH] The website's actual homepage is a byte-for-byte duplicate of the README, and it comes BEFORE "Getting Started" in the nav

`website/sidebars.ts` orders the sidebar `['intro', 'getting-started', ...]`, and
`website/docs/intro.md` has `slug: /` (line 3) — so `intro.md` is literally what a visitor
sees when they land on the docs site root.

`website/docs/intro.md` is a **562-line, word-for-word copy of `README.md`** (confirmed by
direct read — same Quickstart, same BYOM section, same GLiNER section, same LoCoMo benchmark
table, same "Status & maturity" section, same "When not to reach for kremory" section — only
the relative markdown links were rewritten to absolute GitHub URLs). This is not a hub page or
a trimmed landing page — it is the entire README, comprehensiveness and all.

Consequence for the "first 5 minutes" experience: a new visitor to the docs site must scroll
past ~550 lines of reference-depth material (feature-flag tables, benchmark methodology,
observability metric names, ADR citations, "Node.js / MCP — not yet published" caveats)
**before they even reach the "Getting Started" page**, which is the *second* sidebar item and
contains the actual minimal tutorial. The natural onboarding order is inverted: comprehensive
reference-and-marketing material is presented first, the "how do I get a program running"
tutorial second.

Evidence: `website/sidebars.ts:12-14` (`docsSidebar: ['intro', 'getting-started', ...]`),
`website/docs/intro.md:1-562` vs `README.md:1-557` (diffed — content identical modulo link
absolutization).

- **Disposition**: Fixed (doc), together with A2. Rewrote `website/docs/intro.md` from a
  562-line README duplicate down to a ~60-line short landing page: the product pitch (reused
  verbatim from README's own opening framing — tagline, badges, "Local-first" /
  "Memory you can undo" / bi-temporal / BYOM bullets), a one-line install snippet, then a single
  primary "Get started" call-to-action linking to `getting-started.md`, plus a lightweight
  "Learn more" nav list (API Reference / Comparison / Benchmarks) and a short Status/License
  tail. A new visitor no longer scrolls past ~550 lines of reference/marketing material before
  reaching the tutorial.

### A2. [HIGH] Two different, contradicting "smallest working example"s on the first two pages

`intro.md`'s Quickstart (lines 61–116) builds an Ollama provider **directly** via
`autoagents_llm::backends::ollama::Ollama` + `LLMBuilder`, wires a **custom hash-based
embedder** (`DemoEmbedder`), and uses the **Tier 2 `Memory::open(...).with_llm(...)
.with_embedder(...)` builder** with an explicit `.with_model_id(...)` call.

`getting-started.md`'s "smallest working example" (lines 88–114) is a **completely different**
6-line program: `Memory::with_ollama("./agent.db")` (Tier 1), no custom embedder, no explicit
model id, no `LLMBuilder` import.

A newcomer who reads the docs site in nav order sees the FIRST "smallest example" (intro.md,
15+ lines, two custom trait impls, explicit builder chain) and is then told, one click later,
that the ACTUAL smallest example is a different, much simpler 6-line program using an API
(`Memory::with_ollama`) that was never mentioned in the first example. This isn't wrong in
either place individually — `getting-started.md` even explicitly earns its "genuinely the
*first* thing that compiled" claim by describing how it was written from `cargo doc` rustdoc,
independent of this repo's examples — but the two pages were evidently authored independently
and never reconciled against each other as a *reading sequence*. The result reads as two
different projects' onboarding docs stapled together.

Evidence: `website/docs/intro.md:61-116` vs `website/docs/getting-started.md:80-114` (also
`docs/getting-started.md`, identical copy).

- **Disposition**: Fixed (doc), together with A1. Deleted the duplicated Quickstart from
  `website/docs/intro.md` entirely rather than trimming it — nothing on the new landing page is
  a "smallest working example" anymore, only a one-line install snippet. `getting-started.md`'s
  `Memory::with_ollama("./agent.db")` 6-liner is now the ONE canonical first program a reader
  encounters, reachable both from the homepage's CTA and as the second sidebar item. Surfaced a
  related, out-of-scope gap while checking this: `README.md` (the crate-root README, a different
  audience/surface than the docs site) still carries its own, more complex builder-chain
  Quickstart (real Ollama provider built directly via `autoagents_llm`, a custom `DemoEmbedder`,
  explicit `.with_model_id(...)`) that is materially different from `getting-started.md`'s. Per
  the pre-decided scope boundary this is logged as a new LOW finding rather than rewritten — see
  "New finding surfaced during remediation" at the end of this document.

### A3. [HIGH] The acronym "BYOE" means two different things across the docs and the source code itself

Grepped directly (not assumed):

| Site | What BYOE means there |
|---|---|
| `README.md:62`, `README.md:69` ("BYOE — a real consumer calls their embedding backend") | **B**ring **Y**our **O**wn **E**mbedder |
| `docs/getting-started.md:16,96,178` ("BYOM/BYOE", "fully-custom BYOM+BYOE") | Embedder (same as above, paired with BYOM = LLM) |
| `docs/api.md:1430` ("Plug in a custom entity extractor (BYOE); mutually exclusive with `.with_gliner()`") | **B**ring **Y**our **O**wn **E**xtractor |
| `crates/kremory/src/facade/builder.rs:250` (`with_extractor` doc comment: "Provide a custom entity extractor (BYOE — bring your own extractor)") | Extractor — **baked into the source code itself**, not just prose docs |
| `crates/kremory/src/facade/providers.rs:218` ("for the **no-LLM BYOE path**") | Extractor |

This is not a stray typo — it's five independent sites, including two in the actual Rust
source doc comments (which become rustdoc / docs.rs content), consistently using "BYOE" for
"Embedder" in the README/getting-started tier and consistently using it for "Extractor" in the
api.md/source-code tier. A reader who learns "BYOE = embedder" from the front page and later
hits `with_extractor`'s rustdoc entry "(BYOE — bring your own extractor)" on docs.rs will
reasonably think they misread something, or that `with_extractor` is actually about wiring an
embedder. Concrete, load-bearing terminology collision — not a style nit.

- **Disposition**: Fixed (doc), resolved in favour of the source code's meaning. Kept
  "BYOE" = Bring Your Own Extractor everywhere it already meant that (`crates/kremory/src/`:
  `facade/builder.rs:250`, `facade/providers.rs:218`, plus further extractor-context sites in
  `core/intelligence.rs`, `core/ingest/`, `core/provider/chat.rs`, `core/extraction/`, and the
  `kremory-napi` bridge — all verified consistent, no changes needed there). Fixed the 3 prose
  sites that used BYOE to mean "embedder": `README.md:62,69` and `docs/getting-started.md:16,96,178`
  (mirrored into the byte-identical `website/docs/getting-started.md`) — reworded to plain
  language ("a custom embedder", "bring your own chat provider and your own embedder") rather
  than inventing a second acronym. `BYOM` (LLM/model) was already consistent everywhere and was
  left untouched.

### A4. [MEDIUM] `docs/api.md` opens with version-changelog framing instead of "why before how"

The reference's first substantive paragraph (`docs/api.md:3-8`) is: *"the facade gained a
fully-wired dream consolidation phase (§6)... opt-in BM25/FTS5 content recall (§5,
`content-search` — a DEFAULT feature since ADR-078)... The Node/napi binding mirrors the
surface in camelCase (§14)."* This is genuinely useful *provenance* information but it is
change-log register, not orientation — a reader arriving at the "full API reference" for the
first time is handed version-diff prose (what changed since v0.1.3) before a single sentence
of "here is what this reference covers and how it's organised for YOU, the reader, right now."
Every major section thereafter (§2, §3, §5, §6, §13) reads as a flat reference dump organised
by *feature landing order* (chronological, ADR-numbered) rather than by *conceptual
dependency* — which is defensible for a reference doc, but it means the "why would I want
this" framing that IS present (e.g. §5's `.content()` vs `.raw()` latency trade-off table,
§6a's `merge_nogood` rationale) is inconsistently applied: some sections open with the
mechanism (§4 "Basic ingest" just shows code), others open with the trade-off (§5's
`.content()` section is exemplary — states the measured 93.4%/12ms vs 96.7%/112ms trade-off
BEFORE showing code). The doc doesn't have a single answer to "why before how" — it has two
different documents interleaved by feature.

- **Disposition**: Fixed (doc). Added a short orientation paragraph immediately before the
  existing `> **v0.7**` changelog blockquote: states what the reference covers (every builder
  knob / request method / reversibility / feature-flag / Node-binding surface around
  `kremory::Memory`), points a first-time reader at `getting-started.md` first, and names the
  section-ordering convention (feature/adoption order; trade-off-first where one exists,
  mechanism-first otherwise). The changelog content itself is unchanged — it's simply no longer
  the first thing a reader sees. Mirrored into the byte-identical `website/docs/api.md`.

### A5. [MEDIUM] Inconsistent narrative voice between the confident tutorial and the hedge-everything reference

`getting-started.md` is written in a distinctive, first-person "here is what actually happened
when I ran this" voice (*"This is genuinely the first thing that compiled and ran for this
guide"*, *"Every command and every line of output below is real, not smoothed over"*) —
narratively strong, and a genuinely good idea (a captured real transcript builds trust). But
`docs/api.md` and `docs/observability.md` are written in a very different register: dense,
qualifier-heavy, ADR-citing, occasionally defensive prose (e.g. `docs/api.md:1181` "which is
how a config-mismatch has produced a bogus benchmark number in the past" — an aside about an
internal historical incident that means nothing to an external reader and reads as an internal
postmortem leaking into public API docs). A reader moving from getting-started.md (personable,
narrative, first-person) to api.md (dense, internal-incident-citing, ADR-numbered) experiences
a genuine tone whiplash — it doesn't read as one voice describing one product.

Evidence of the internal-incident leak: `docs/api.md:1176-1182` ("Introspection — reading back
the ACTIVE config" section justifies its existence by referencing "how a config-mismatch has
produced a bogus benchmark number in the past" — a fact meaningful only to the maintainers,
not to a consumer deciding whether to call `search_config()`).

- **Disposition**: Fixed (doc + code, prose-only). Rewrote the `docs/api.md` §10 "Introspection"
  aside to frame `search_config()`'s value from the consumer's perspective — avoiding drift
  between what a health-check endpoint / debug log / benchmark harness reports and what the
  recall path actually uses — rather than citing the internal historical incident. Mirrored into
  `website/docs/api.md`. Also fixed the byte-identical phrasing in its actual source, the rustdoc
  doc comment on `Memory::search_config` at `crates/kremory/src/facade/mod.rs:689-693` (which
  ships to docs.rs and is the origin of the wording docs/api.md was summarising) — same defect,
  same fix, prose-only with no signature change, so bundled at zero extra blast radius even
  though the finding named only the prose doc.

### A6. [LOW] `docs/comparison.md` visibly two-tiered in trust, and says so loudly at the top — good honesty, but the doc itself is stale-by-design

Not really a *writing quality* issue — the doc explicitly flags (line 32-38) that competitor
cells are unverified since 2026-05-22 while kremory cells are marked `✏️` and updated. This is
commendably honest self-disclosure. Flagging as LOW only because a doc that is ~3.5 months
stale on more than half its content, sitting in the "Reference" nav category presented with
equal visual weight to `benchmarks.md` (which IS current), risks a reader skimming the table
and not reading the caveat block above it (a common failure mode for comparison tables —
readers screenshot/quote the table, not the caveat). Consider a per-column "last verified"
date directly in the table header row rather than only in prose above it.

- **Disposition**: Fixed (doc), per the finding's own suggested mechanism. Added a per-column
  "last verified" marker directly in the comparison table's header row — `**kremory** (updated
  2026-09-04)` for the kremory column, `(unverified since 2026-05-22)` for every competitor
  column — so the staleness signal survives a reader skimming only the table. The existing prose
  caveat block above the table is unchanged (still useful standalone context). Mirrored into
  `website/docs/comparison.md` (whose only difference from `docs/comparison.md` is the absence
  of the aidocs frontmatter block; body kept in sync).

### A7. [LOW] `getting-started.md` "Next steps" doesn't loop back to `intro.md` at all, and doesn't warn the reader that `intro.md`/README covers overlapping ground

Given A1/A2 above, a reader who starts at `getting-started.md` (e.g. arrives via a direct link,
skipping the homepage) has no signal that `intro.md` exists or that it re-covers the same
Quickstart ground with a different example. Not a broken link — a missing cross-reference that
would materially help readers navigate around the duplication in A1/A2 until it's fixed.

- **Disposition**: Resolved by A1/A2 (no additional change needed). Re-assessed per the
  pre-decided instruction: `intro.md` is now a short, clearly-distinct landing page with no
  Quickstart of its own, so the specific harm A7 flagged — a reader navigating around
  undisclosed *duplication* — no longer applies, because there is no more duplication to warn
  about. Did not add a cross-reference from `getting-started.md` back to `intro.md`; the only
  honest thing left to say would be "the homepage exists," which `website/sidebars.ts` already
  makes true by construction (it's the first nav item), so a manual cross-link would be
  restating the site's own navigation rather than adding information.

### Summary table — Part A

| # | Severity | Dimension | Finding |
|---|---|---|---|
| A1 | HIGH | Narrative/pacing | Homepage = full README dump, placed before the actual tutorial in nav order |
| A2 | HIGH | Consistency / first-5-min | Two different, unreconciled "smallest working examples" on the first two pages |
| A3 | HIGH | Consistency (terminology) | "BYOE" means Embedder in README/getting-started, Extractor in api.md + the source code itself |
| A4 | MEDIUM | Completeness of framing | api.md opens with changelog prose, not orientation; why-before-how applied inconsistently across sections |
| A5 | MEDIUM | Consistency (tone) | Tutorial voice (personal, narrative) vs reference voice (dense, internal-incident-citing) never reconciled |
| A6 | LOW | Consistency | comparison.md's staleness caveat is easy to miss if only the table is skimmed |
| A7 | LOW | Narrative/pacing | No cross-reference from getting-started.md warning readers about the intro.md/README overlap |

---

## Part B — API design vs the Rust API Guidelines

Guidelines fetched live from `rust-lang.github.io/api-guidelines/checklist.html` and
`.../type-safety.html` (2026-09-07), not from training memory, per the checklist's own C-codes.
Reviewed: `crates/kremory/src/facade/builder.rs` (the full 1586-line file) + the `MemoryBuilder`
surface documented in `docs/api.md` + a targeted grep of `Debug`/`Error` derive coverage and
crate metadata.

### B1. [HIGH] `Memory`, the primary consumer-facing type, does not implement `Debug` — violates C-DEBUG

C-DEBUG: *"All public types implement Debug."* Grepped directly:
`crates/kremory/src/facade/mod.rs:509-510` shows `#[derive(Clone)] pub struct Memory { ... }`
— **no `Debug` derive, and no manual `impl Debug for Memory` anywhere in the file** (confirmed
by `grep -n "impl.*Debug.*for Memory"` returning nothing). `MemoryBuilder<L, E>`
(`facade/builder.rs:30`) likewise has no `Debug` derive.

This is understandable mechanically — both types hold `Arc<dyn ChatProvider>` /
`Arc<dyn GraphHandle>` trait objects that don't implement `Debug` themselves, so a naive
`#[derive(Debug)]` wouldn't compile — but the guideline's own remedy for exactly this case is
a hand-written `impl Debug` that prints what CAN be shown (path, namespace, whether an LLM/dream_llm/sink is configured) and elides the rest, which is standard practice for library
types wrapping trait objects (e.g. how `tokio::runtime::Runtime` or `reqwest::Client` handle
it). Concretely, today, `println!("{:?}", mem)` or `dbg!(mem)` — the first thing most Rust
developers reach for when confused — does not compile for the type users interact with most.
This is a real ergonomic and debuggability gap, not a cosmetic one.

- **Disposition**: Fixed (code). Added a hand-written `impl std::fmt::Debug for Memory`
  (`crates/kremory/src/facade/mod.rs`) and `impl<L, E> std::fmt::Debug for MemoryBuilder<L, E>`
  (`crates/kremory/src/facade/builder.rs`, no `L: Debug` / `E: Debug` bound needed since the
  phantom type-state markers are never printed). Both print presence-only for the opaque
  `Arc<dyn Trait>` fields (`llm_configured`, `embedder_configured`, `event_sink_configured`,
  etc.), verbatim for everything else (`model_id`, `default_namespace`, `embedding_dim`,
  `dream_schedule`, `await_extraction`, `await_extraction_timeout`, …), and both call
  `.finish_non_exhaustive()` — the same shape `tokio::runtime::Runtime` / `reqwest::Client` use,
  per the finding's own cited precedent. Added 2 new tests
  (`memory_debug_reports_configured_state`, `memory_builder_debug_reports_unconfigured_state` in
  `crates/kremory/tests/it/facade_open.rs`) asserting `format!("{:?}", ...)` now compiles and
  reports the expected configured/unconfigured state — directly verifying the finding's own
  claim (that `println!`/`dbg!` didn't compile before) rather than just asserting the impl
  exists.

### B2. [MEDIUM] Inconsistent "does this mutation need `.execute()`?" rule across the facade

The guidelines' Predictability section (C-METHOD, and the general principle that operator/API
surprise should be minimized) is the closest match for this finding, though it isn't one exact
checklist line — it's a self-inflicted inconsistency worth flagging on its own terms.

Reading `docs/api.md` end to end, the facade has TWO terminal conventions for request builders:

| Builder | Terminal |
|---|---|
| `mem.remember(...)` | bare `.await` |
| `mem.recall(...)` | bare `.await` |
| `mem.dream(...)` | bare `.await` |
| `mem.forget()` | **`.execute().await`** |
| `mem.edit_entity(...)` | **`.execute().await`** |
| `mem.delete_entity(...)` / `.delete_fact(...)` | **`.execute().await`** |
| `mem.supersede(...)` | **`.execute().await`** |
| `mem.undo(...)` | **`.execute().await`** |

The docs give an explicit, sensible-sounding rationale for `forget()` specifically
(`docs/api.md:924-925`: *"forget() returns a builder; the destructive operation only fires on
.execute(). This explicit terminal makes the intent visible in code review."*) — but that
rule is not applied consistently. **`dream()` is arguably the MOST consequential call in the
whole API** — by the docs' own description (§6: *"All ops default ON... Committing by default
is safe because every destructive mutation is reversible"*) it commits entity merges, fact
archival, and a supersession sweep across potentially the whole graph — yet it uses bare
`.await`, exactly like the non-destructive `recall()`. Meanwhile `undo()` — which *reverses* a
mutation, i.e. is the safety net, not the risk — requires the same explicit `.execute()` as
`forget()`. A newcomer cannot predict, from the shape of the API alone, which calls need
`.execute()` and which don't; they have to memorize a table (the one above), because the
stated rule ("destructive operations get an explicit terminal for code-review visibility")
does not actually correlate with which calls need it in practice.

- **Disposition**: Fixed (code + doc), breaking change. `DreamRequest` now requires an explicit
  `.execute()` terminal, matching `forget()`/`undo()`/etc. exactly: removed its `IntoFuture` impl,
  renamed the private `execute_blocking` to `pub async fn execute`, and applied the same
  `#[must_use = "DreamRequest must call .execute() to run"]` on both the struct declaration and
  `Memory::dream()` that every other destructive-terminal builder in the file already carries.
  `.fire_and_forget()` is unaffected — it already had its own explicit terminal.
  Updated every real call site in the repo: `crates/kremory/src/facade/mod.rs` (2 doctest sites),
  `crates/kremory-napi/src/lib.rs`, `crates/kremory-mcp/src/handlers.rs`, 16 integration test
  files, `README.md`'s "Reversible dream" example, and `docs/api.md`'s 6 code-fenced `dream()`
  examples (mirrored into `website/docs/api.md`). Every Rust call site was found by letting
  `cargo check --workspace --all-features --all-targets` enumerate the breakage via compile
  errors (`DreamRequest<'_> is not a future`) rather than by grep — the compiler is the
  authoritative list of what actually needed fixing, and it caught 29 real sites across 3 crates.
  Napi parity impact: this is one genuinely new tracked symbol (`DreamRequest::execute`); added a
  skip-list entry mirroring the existing `ForgetRequest::execute` sibling (same
  implementation-detail rationale) and raised the sanity cap 91 → 92, updating the lead comment /
  history note / assert in `crates/kremory-napi/tests/api_parity.rs` per the file's own
  documented convention for this exact kind of raise. Verified: `cargo test -p kremory-napi --test
  api_parity` 7/7 green, cap now sitting AT 92/92.

### B3. [MEDIUM] Type-state markers `NoLlm`/`WithLlm` vs `NoEmb`/`WithEmb` break C-WORD-ORDER / naming-consistency expectations

The two phantom type-state parameters are asymmetrically abbreviated: `NoLlm`/`WithLlm` spell
out "Llm" in full, while the embedder pair is truncated to `NoEmb`/`WithEmb` rather than
`NoEmbedder`/`WithEmbedder`. Both are re-exported at the crate root
(`crates/kremory/src/lib.rs`: `pub use facade::{..., NoEmb, NoLlm, ..., WithEmb, WithLlm,
...};`), so this is public, permanent API surface, not an internal detail — and the asymmetry
(one name abbreviated, its sibling not) is exactly the kind of "names use a consistent word
[and form] order" (C-WORD-ORDER) issue the guidelines call out, on a pair of types a consumer
is fairly likely to encounter if they ever write a generic function over `MemoryBuilder<L, E>`.

- **Disposition**: Fixed (code), breaking rename. `NoEmb`/`WithEmb` → `NoEmbedder`/`WithEmbedder`
  everywhere: the marker struct declarations (`crates/kremory/src/facade/mod.rs`), every generic
  usage across the builder's type-state impl blocks (`crates/kremory/src/facade/builder.rs`), the
  crate-root re-export (`crates/kremory/src/lib.rs`), and 2 integration test files
  (`facade_open.rs`, `facade_event_sink.rs`). Also updated the doc-comment references in
  `crates/kremory-napi/src/lib.rs` (prose only — the napi crate has no Rust-level generic surface
  of its own naming these types). `NoLlm`/`WithLlm` were already correct and untouched. Verified
  via `cargo check --workspace --all-features --all-targets` (clean) and a word-boundary grep
  confirming zero remaining `NoEmb`/`WithEmb` occurrences anywhere in the tree.

### B4. [LOW] Ten `with_X_enabled(bool)` boolean setters coexist with one honest tri-state enum (`CrossEpisodeMode`) for the same *class* of decision, with no stated reason for the split

`docs/api.md §13` documents `with_episode_dense_enabled(bool)`, `with_fact_dense_enabled(bool)`,
`with_embed_task_prefix_enabled(bool)`, `with_contradiction_detection_enabled(bool)`, and
others — plain boolean toggles. Elsewhere in the exact same builder, the project deliberately
chose NOT to do this for cross-episode merging: `docs/api.md:706-707` explicitly says *"prefer
the honest tri-state `CrossEpisodeMode` over toggling the two coupled raw bools
(`include_cross_episode_merges` + `cross_episode_dry_run`)"* — i.e. the maintainers have
already identified, and fixed, the C-CUSTOM-TYPE anti-pattern ("arguments convey meaning
through types, not bool") for exactly one knob, and left roughly ten structurally similar
on/off toggles as plain bools. Each individual bool setter is unambiguous by its own name
(`with_fact_dense_enabled(true)` is readable), so this is genuinely LOW severity as a
usability problem in practice — but it means the codebase has silently inconsistent taste
about its own documented anti-pattern, one it clearly knows how to avoid.

- **Disposition**: Fixed (doc only — no code change, per the pre-decided call that a ~10-method
  breaking rename is disproportionate for a LOW-severity taste inconsistency). Added a short
  note in `docs/api.md` §13 (mirrored into `website/docs/api.md`), immediately after the
  "Advanced tuning knobs" table, explaining the actual design rule: a plain `bool` stays a `bool`
  when it toggles ONE independent, orthogonal decision; `CrossEpisodeMode` exists specifically
  because cross-episode merging is controlled by TWO COUPLED raw bools whose combinations can be
  ambiguous to read at a call site, and the tri-state enum makes every valid combination a single
  self-explaining value. This makes the split legible as a deliberate distinction rather than
  silent inconsistency, without touching the public API.

### B5. [LOW] `.with_ollama_at_model` free function vs `.with_ollama_at` inherent method — inconsistency the docs already flag on themselves

`README.md:263-265` (and the corresponding `docs/api.md`) states directly: *"`with_ollama_at_model`
lives in `kremory::facade::providers` (a free function, not an inherent `Memory::` method
like its sibling `with_ollama_at`)"* — i.e. two near-identically-named functions differ in
whether they're inherent `Memory::` associated functions or free functions in a submodule, for
no reason apparent to the consumer other than implementation history. Flagging LOW only
because the docs are already transparent about it rather than hiding it — but the underlying
API inconsistency is real and a rename/re-export to make both inherent (or both free) would be
a small, low-risk fix.

- **Disposition**: Fixed (code + doc). Added a thin inherent
  `Memory::with_ollama_at_model(url, model, path)` (`crates/kremory/src/facade/mod.rs`)
  delegating to the existing `facade::providers::with_ollama_at_model` free function, which
  remains callable unchanged (additive; the free function is not deprecated). Updated
  `README.md`'s "Prefer a smaller footprint?" example to call `Memory::with_ollama_at_model(...)`
  as the primary form, with prose noting the free function still reaches the same code for
  anyone already depending on it. This closes out a deferral explicitly left open by the earlier
  correctness-review pass (`phase1-findings.md`'s disposition for the equivalent finding there
  said adding this wrapper "remains the maintainer's call if wanted, and is not required to make
  the doc correct") — it is now built. Added 2 new tests
  (`with_ollama_at_model_accepts_custom_url_and_model`,
  `with_ollama_at_model_defaults_when_model_is_none` in
  `crates/kremory/tests/it/facade_tier1_shortcuts.rs`), mirroring the existing
  `with_ollama_at_accepts_custom_url` sibling test's shape (no live Ollama needed — construction
  doesn't touch the network). Also added a row to `crates/kremory-mcp/surface-decisions.toml`
  (`with_ollama_at_model = { deferred = "server-owned construction" }`, matching its
  `with_ollama_at` sibling) — the MCP surface-decision gate fails on any new public `Memory`
  operation with no recorded decision, and this method is new public surface.

### B6. What the API gets RIGHT (for balance — not everything is a finding)

- **Consuming builder shape matches C-BUILDER's "consuming builder" variant exactly**: every
  setter takes and returns owned `self`, terminal is reached via `IntoFuture` (`.await`) rather
  than a bare `.build()` — a deliberate, well-executed choice, and the `#[must_use = "..."]`
  attribute on `MemoryBuilder` (`builder.rs:29`) gives a genuinely helpful compiler nudge
  ("MemoryBuilder must be configured with .with_llm() AND .with_embedder() before .await").
- **Type-state compile-time enforcement of the two required setters** is real, tested (the
  `compile_fail` doctest in `docs/api.md:92-95`), and matches the stated design goal exactly —
  this is advanced Rust used for a genuinely good reason (catch a real class of misconfiguration
  at compile time), not decoration.
- **`Namespace` (the most commonly-constructed value type) derives the full common-trait set**
  — `Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize` (`memory/types.rs:35`) —
  satisfying C-COMMON-TRAITS and C-SERDE cleanly.
- **Error types use `thiserror` with `#[derive(Debug, Error)]` and `#[from]` conversions**
  (`core/error.rs:130`, `memory/types.rs:1436`) — satisfies C-GOOD-ERR's shape expectations
  (meaningful `Display`, composable via `?`, `Send + Sync + 'static` by construction).
- **`composable-knobs-over-strategy-enum` shape is used correctly for the recall-tuning
  surface** (`with_content_stream_weight`, `with_rrf_k`, `with_proximity_weight`, etc. — each
  an independent, optional override rather than one big config-struct literal) — matches
  C-BUILDER's spirit of incremental configuration and the project's own stated naming/design
  rules.
- **Cargo.toml metadata is complete** — `description`, `license`, `repository`, `homepage`,
  `readme`, `keywords`, `categories` are all present (C-METADATA satisfied); the `homepage`
  field even has an honest inline comment explaining why it points at GitHub rather than a
  registered-but-unserved domain, rather than silently shipping a dead link.
- **No stray `get_` prefixed methods** found in the facade (grepped directly) — C-GETTER
  respected.

### Summary table — Part B

| # | Severity | Guideline | Finding |
|---|---|---|---|
| B1 | HIGH | C-DEBUG | `Memory` and `MemoryBuilder` (the two most-used public types) implement no `Debug` |
| B2 | MEDIUM | Predictability (no single C-code, general principle) | Inconsistent rule for which builders need explicit `.execute()` — `dream()` (highly destructive) doesn't need it, `undo()` (restorative) does |
| B3 | MEDIUM | C-WORD-ORDER | `NoEmb`/`WithEmb` abbreviated, `NoLlm`/`WithLlm` not — asymmetric naming on a public type pair |
| B4 | LOW | C-CUSTOM-TYPE | ~10 plain-`bool` setters coexist with one deliberately-fixed tri-state enum for the same class of problem |
| B5 | LOW | Predictability | `with_ollama_at_model` (free fn) vs `with_ollama_at` (inherent method) inconsistency, already self-flagged in docs |
| — | (positive) | C-BUILDER, C-COMMON-TRAITS, C-SERDE, C-GOOD-ERR, C-METADATA, C-GETTER | All satisfied cleanly — see B6 |

---

## Part C — Docusaurus customization inventory (report only)

Checked what is ALREADY configured vs left as `create-docusaurus` scaffolding defaults, based
on `website/docusaurus.config.ts`, `website/package.json`, `website/sidebars.ts`, and a direct
listing of `website/src/` + `website/static/`.

### Customized (real, kremory-specific work)

- **Site identity**: `title: 'kremory'`, `tagline: 'The SQLite of agent memory'`,
  `organizationName: 'kgentic'`, `projectName: 'kremory'` — all set correctly for this project.
- **Navbar + footer links** — wired to the real GitHub repo, crates.io page, and docs.rs, plus
  internal links to Getting Started / API Reference (`docusaurus.config.ts:73-124`).
- **Manual sidebar** (`sidebars.ts`) — hand-authored category structure ("API Reference" /
  "Operations" / "Reference") rather than the auto-generated-from-folder-structure default,
  with an explicit code comment tying it back to a migration plan (RULE-009: 100% of existing
  docs reachable from nav).
- **`editUrl`** points at the real GitHub tree path for the docs, so the "Edit this page" link
  works.
- **`blog: false`** — deliberately disabled the blog plugin (docs-only site), not left as
  scaffolding default (default scaffold ships blog ON).
- **`colorMode.respectPrefersColourScheme: true`** — a real, non-default choice (stock
  scaffold does not set this).
- **Homepage feature copy** (`src/components/HomepageFeatures/index.tsx`) — the three feature
  blurbs ("Local-first", "Memory you can undo", "Bring your own model") are genuine kremory
  copy, not the stock "Easy to Use / Focus on What Matters / Powered by React" scaffold text.
- **Prism syntax-highlighting theme pair** (`prismThemes.github` / `prismThemes.dracula`) is
  explicitly set rather than left implicit.

### NOT customized — still stock `create-docusaurus` scaffolding

- **`custom.css`** (`src/css/custom.css`, 30 lines) is **100% the literal default template
  file** — the exact stock Infima green/teal palette (`--ifm-color-primary: #2e8555` light /
  `#25c2a0` dark) with the placeholder comments ("You can override the default Infima variables
  here") still present, word for word. There is zero kremory-specific brand color anywhere in
  the site's CSS.
- **Homepage illustration SVGs** — `undraw_docusaurus_mountain.svg`,
  `undraw_docusaurus_tree.svg`, `undraw_docusaurus_react.svg` are the **stock unDraw
  illustrations Docusaurus ships in its own init template** (their filenames literally say
  "docusaurus"), reused as-is under the kremory feature headings — a mountain/tree/React-logo
  illustration set has no visual connection to a Rust bi-temporal knowledge-graph library.
- **`logo.svg` / `favicon.ico`** — almost certainly the unmodified default Docusaurus logo:
  directly inspected the SVG source and its fill colors (`#3ECC5F` etc.) match Docusaurus's own
  known brand palette, not any kremory-specific mark. (Stated with the confidence a direct
  color-value check supports, not a byte-for-byte hash comparison — worth a maintainer glance
  to confirm before assuming.)
- **`static/img/docusaurus-social-card.jpg`** and `static/img/docusaurus.png` — present,
  unrenamed, unreplaced; `docusaurus.config.ts:62-63` even has an inline comment
  acknowledging this: *"Replace with a real social card image when the site is actually
  deployed."*
- **No search** — no Algolia config, no local-search plugin in `docusaurus.config.ts` or
  `package.json` dependencies. A site-wide text search box does not exist today.
- **No versioning** — no `versions.json`, no `versioned_docs/` — appropriate for a pre-1.0
  single-tracked-version project, but worth naming explicitly since the guideline task asked
  for it: this is scaffold-default (versioning was never turned on), not a deliberate
  "we decided against it" choice recorded anywhere.
- **No i18n** beyond the single default `en` locale (`i18n: { defaultLocale: 'en', locales:
  ['en'] }`, `docusaurus.config.ts:39-42`) — this is actually the create-docusaurus default
  shape, just confirmed unexpanded.
- **`url: 'https://kremory.dev'` / production deploy config** is explicitly a placeholder —
  `docusaurus.config.ts:17-19` comments state *"this site is NOT deployed as part of this
  change... deploy target/DNS is a separate, maintainer-gated phase."* Consistent with the
  root `Cargo.toml`'s own comment that `kremory.dev` is registered but serves no DNS record yet
  (cross-referenced, not re-derived: `crates/kremory/Cargo.toml`'s `homepage` field comment
  says the same thing independently).

**Net picture for the maintainer**: the *content* layer of the site (nav structure, page
copy, doc migration) has real, deliberate work behind it. The *visual identity* layer (colors,
illustrations, logo, social card) is entirely stock scaffolding — the site currently looks,
visually, like an unbranded `create-docusaurus` starter with kremory's words typed into it.
That gap is invisible until you go looking for it, because Infima's default green/teal palette
is inoffensive and doesn't look "obviously wrong" the way a broken link does.

---

## Overall verdict

**Documentation quality: the individual documents are unusually rigorous and honest (real
transcripts, dated benchmark caveats, explicit "here's what's stale" flags) — but the SET of
documents was never edited as a reading sequence, and that's the dominant defect.** A newcomer
following the nav in order hits a 562-line README-duplicate before the tutorial, then hits a
second, different "smallest example" a page later, then hits a genuine terminology collision
(BYOE = embedder vs BYOE = extractor) baked into the actual source code doc comments. None of
these are "the docs are wrong" — every individual sentence checked out — they are "the docs
were assembled from several honestly-good but independently-written pieces, and nobody read
them start to finish as a single onboarding journey."

**API design: genuinely well-considered where it counts (compile-time-enforced required
setters, consuming-builder shape matches the canonical guideline pattern exactly, composable
knobs over a strategy enum, solid error types) — with two real gaps a newcomer will hit
quickly**: no `Debug` on the main `Memory`/`MemoryBuilder` types (breaks the reflex of
`dbg!(mem)`), and an unpredictable rule for which operations need `.execute()` (the docs claim
"destructive operations get an explicit terminal" but the single most destructive call,
`dream()`, doesn't have one, while the reversal call, `undo()`, does).

**Findings count:**

| Part | HIGH | MEDIUM | LOW |
|---|---|---|---|
| A (docs quality) | 3 | 2 | 2 |
| B (API design) | 1 | 2 | 2 |
| **Total** | **4** | **4** | **4** |

**If only one thing could be fixed:** Reconcile A1 + A2 together as a single fix — collapse
`intro.md` back down to a short, distinct landing page (product pitch + one link to Getting
Started), and delete or rewrite its duplicated Quickstart so there is exactly ONE canonical
"first program a newcomer runs," used consistently by the homepage AND getting-started.md.
This one change fixes the worst pacing problem (A1), the worst consistency problem (A2), and
removes the site's most confusing early-reading experience — and it's also the cheapest fix on
the list (it's an edit to one already-over-long file, not new design work), unlike B1/B2 which
require actual Rust code changes.

---

## Remediation pass — 2026-09-07

All 12 findings (A1-A7, B1-B5) have a **Disposition** recorded inline above. Summary: **11
Fixed, 1 Resolved-by-a-sibling-fix (A7), 0 NOT-FIXED.** B2 and B3 are breaking Rust API changes
(shipped as `0.7.1`, prepared but not yet published — see `CLAUDE.md`'s standing `cargo publish`
HITL gate); every other finding is additive or doc-only.

Quality gate, run against the full remediation diff:

- `cargo build --workspace` — clean.
- `cargo clippy -p kremory -p kremory-napi --all-targets --all-features` — zero warnings.
- `cargo nextest run -p kremory -p kremory-mcp --features content-search,test-utils` —
  **1800/1800 passed, 6 skipped** (grew from the pre-existing 1796 by exactly the 4 new tests
  this pass added: 2 for B1's `Debug` impls, 2 for B5's `with_ollama_at_model` inherent method).
- `cargo test -p kremory-napi --test api_parity` — 7/7 green; `parity-skip.toml` sanity cap
  raised 91 → 92 for exactly one new entry (`DreamRequest::execute`, B2's new method), following
  the file's own documented raise convention; sitting AT the cap.
- `cargo test --doc -p kremory-doc-examples` (after `python3
  crates/kremory-doc-examples/generate_docs.py`) — **24/25 passed.** The one failure
  (`docs_observability_md` line 61, `metrics_exporter_prometheus` unresolved) is a **pre-existing,
  unrelated** gap: `docs/observability.md` (a file this remediation pass never touched) contains a
  code fence referencing a crate the harness's own `Cargo.toml` never lists as a dependency. Not
  fixed here — out of scope for these 12 findings; classified as a FIXABLE BUG in
  `crates/kremory-doc-examples/Cargo.toml` (add `metrics-exporter-prometheus` as a dev dependency)
  for whoever next touches `docs/observability.md`.

### New finding surfaced during remediation (LOW, not fixed — logged per the pre-decided scope
boundary)

**README.md's Quickstart still contradicts getting-started.md's "smallest working example",
the same shape as the fixed A2, just on a different pair of pages.** `README.md`'s Quickstart
(lines 56-117) is the same complex example the pre-fix `website/docs/intro.md` carried — a real
Ollama provider built directly via `autoagents_llm::backends::ollama::Ollama` + `LLMBuilder`, a
custom hash-based `DemoEmbedder`, the Tier 2 builder with an explicit `.with_model_id(...)` call.
`docs/getting-started.md`'s smallest example is the 6-line `Memory::with_ollama("./agent.db")`
Tier-1 shortcut. A reader who reads GitHub's README first and the docs site second (or vice
versa) hits the same "two different smallest examples" experience A2 fixed for the docs-site
homepage.

Per the pre-decided scope for this remediation pass, README is the crate-root README — a
different audience (GitHub/crates.io browsers, not docs-site visitors) and a higher-blast-radius
surface than the two files A1/A2 already touched, so it was left as-is rather than rewritten.
Flagging here as a genuinely new LOW finding for a future pass: either (a) trim README's
Quickstart to the same 6-line Tier-1 example `getting-started.md` uses, with the fuller
custom-embedder example moved to `docs/api.md` §2 where the "customizing the LLM/embedder"
content already lives, or (b) explicitly note in README that a simpler path exists and point at
`getting-started.md`. Not actioned in this pass.
