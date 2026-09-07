# kremory-doc-examples

Internal, `publish = false` harness. Phase 0 of
`docs/specs/public-docs-and-api-surface-audit/` (RULE-001, RULE-002; tasks
T0.1-T0.4). Compiles every fenced ` ```rust ` block in `docs/*.md` +
`README.md` against the real `kremory` crate, so drift between what's
documented and what ships becomes a compile error instead of a support
ticket.

This crate is not shipped anywhere and is not part of `kremory`'s public API
surface.

## How to run

```sh
python3 crates/kremory-doc-examples/generate_docs.py   # regenerate ./generated/*.md
cargo test --doc -p kremory-doc-examples                # compile them all
```

`./generated/` is gitignored (build artifact, regenerated on demand — see
`.gitignore` in this directory).

## Mechanism (T0.1 — spiked before building the rest)

Two wiring options existed per `plan.md`'s Phase 0 note:

- **(a)** bare `rustdoc --test <file>.md --extern kremory=... -L ...`
- **(b)** a tiny `#[doc = include_str!(...)]` module in a **test-only crate**,
  exercised via `cargo test --doc`

**(b) won**, confirmed by a real spike against a 2-block fragment lifted from
`docs/api.md`'s own Quickstart section (a throwaway crate outside this repo,
depending on `kremory` via a `path` dependency — so nothing under
`crates/kremory/src/` was ever touched): `kremory::` imports resolved
cleanly, and a follow-up spike confirmed `no_run` (compile-only, never
executes) and `compile_fail` (must NOT compile) behave exactly per rustdoc's
documented semantics. (a) would have required hand-computing `--extern`
flags for every crate a doc snippet happens to reference (`anyhow`, `chrono`,
`tokio`, `async-trait`, `autoagents-llm`, `metrics`, `metrics-util`, ...);
(b) gets all of that for free from Cargo's own dependency resolution, which
is exactly why the plan called it "the more common crate idiom."

This crate exists **because** of that constraint — "test-only crate" in the
plan's own wording is precisely what lets this live outside
`crates/kremory/src/`, honoring the Phase 0 rule that this pass is read-only
against the crate under audit.

## Extraction + generation (T0.2)

`generate_docs.py` walks `docs/*.md` + `README.md`, extracts every fenced
` ```rust ` block, and groups them into compilation units. See the module
docstring and inline comments in that file for the mechanics; the load-bearing
design decisions are below.

### Grouping granularity: `##` only, not `##`/`###`

RULE-001's literal text ("blocks under one `##`/`###` heading... share one
compilation unit") reads as splitting on *either* level. The harness instead
groups on `##` only, treating `###` purely as a label. Reason: a real
cross-`###`-subsection dependency exists in `docs/api.md` Section 6a — `###
SEE` binds `history`; `### UNDO`, a separate `###` two headings later under
the *same* `##`, reads `history.first()` with no local definition. Splitting
at `###` produces a false "cannot find value `history`" failure that is an
artifact of over-fine grouping, not a real doc defect. The trade-off, stated
plainly: some `##` sections in `docs/api.md` (Section 5 in particular) end up
as one large 9-block unit, so a single compile error there is attributed to
the whole section rather than to one sub-block until a human reads the
compiler's line-mapped error.

### The placeholder prelude (RULE-001's Blocker-1 requirement)

RULE-001 mandates, at minimum: `my_llm: Arc<dyn ChatProvider>`, `my_embedder:
Arc<dyn DynEmbeddingProvider>`, and a `MySink` unit struct implementing
`EnrichmentEventSink`. Empirically, against the real docs, that list was far
from exhaustive. The full prelude (`PRELUDE_HEADER` + `PRELUDE_BODY_ONCE` +
`PRELUDE_REFRESH_LINES` in `generate_docs.py`) also provides:

| Name | Why |
|---|---|
| `mem: Memory` | Almost every section after §1 (Quickstart) assumes a `Memory` already exists and never constructs one locally. |
| `llm`, `emb`, `l`, `e` | Short-name aliases used in `docs/api.md`'s multi-tenant example and `docs/error-handling-policy.md`'s facade-error example. |
| `ns: Namespace` | Used bare in several sections without a local `let`. |
| `x`, `y`: `DateTime<Utc>` | `docs/api.md` §11 ("the second clock") uses these as generic timestamp stand-ins. |
| `turn_text`, `question`: `&str` | `docs/api.md` §5.1's session-expansion recipe. |
| `fact_id: i64` | `docs/api.md` §6a's direct-mutation examples (`delete_fact`/`supersede`). |
| `MyCustomChatProvider`, `MyCustomEmbeddingProvider` | `docs/observability.md` names these types via `::new()` without ever showing a body — real, delegating impls wrapping `MockChatProvider`/`NullEmbeddingProvider` so the surrounding `.with_llm(Arc::new(MyCustomChatProvider::new()))` call type-checks. |

**Deliberate scope limit:** `MyEmbedder` (README's BYOM section) and
`MyGraphBackend` (`docs/api.md` §10) are **not** in the prelude — both are
defined inline, in full, by their own doc snippet. `MyGraphBackend`'s inline
`impl GraphHandle` is empty (`// implement all required methods`), which is
a genuine finding (see the audit table below), not a prelude gap.

### Why standard placeholders are re-emitted before every original block, not `macro_rules!`'d once

The first cut used a `macro_rules!` block invoked once per original snippet to
give each one "fresh" `my_llm`/`llm`/`ns`/etc. bindings — so that reusing the
same placeholder name across multiple **originally-independent** snippets,
now concatenated into one compilation unit, never trips a "value moved" error
that would be a harness artifact rather than a real doc-vs-code defect. That
broke on the very first real run: Rust macro hygiene makes `let` bindings
created *inside* a `macro_rules!` invocation invisible to code *outside* that
specific invocation — `cannot find value my_llm... not accessible due to
macro hygiene`. Fixed by dropping the macro and re-emitting the same lines as
plain, repeated text (`PRELUDE_REFRESH_LINES`) before every original block.

### Deduplicating `use` imports across concatenated blocks

Every doc fragment is written to read standalone, so many independently
re-import the same common names (`use kremory::Namespace;` appears in several
sibling snippets under `docs/api.md` §3, for instance). Rust forbids two
local `use` bindings of the same name in one scope — even when both resolve
to the identical item — so naively concatenating such blocks produces a false
`E0252` the moment two sibling snippets, now sharing one function body, both
import the same name. `dedupe_block_code()` in `generate_docs.py` tracks
names already in scope (seeded from the prelude's own blanket imports) and
elides a name from a later `use` if it's already there, per-unit. This
resolved every case in the current docs (`Namespace`, `Memory`, `GraphHandle`
each collided at least once) without deleting any doc content.

### Marker vocabulary (T0.2 §3 of the task)

A block is recognized as **not required to compile** if:

- its fence info string contains `compile_fail` (isolated into its own
  compilation unit, never merged with siblings, so a deliberately-broken
  example can't corrupt a sibling's pass/fail signal) or `ignore`
  (rustdoc's own "skip entirely" marker), or
- the prose since the enclosing heading contains "illustrative" or
  "pseudo-code" (case-insensitive).

An explicit `no_run` marker does **not** exempt a block from compiling — per
RULE-001, "compile != run": Phase 0 is a type-check only, and every
"normal"/`no_run` unit is forced to `no_run` in the *generated* fence
regardless of the original marker, so nothing in this harness ever makes a
network call or touches a real filesystem path beyond compiling. As of this
writing, **zero** of the 69 blocks in scope carry any of these markers — the
vocabulary is implemented and will fire the moment one is added, but every
current failure is genuinely unmarked.

### RULE-002 vacuous-pass guard (T0.3)

`generate_docs.py`'s `main()` asserts the total Rust-block extraction count
is `>= 60` and exits non-zero (loudly) if not — this has bitten the repo
twice before (`consistency_check`, the ADR-integrity guard's own scan set),
per the spec.

## Known, deliberate gaps (not papered over)

- **`metrics_exporter_prometheus`** — `docs/observability.md`'s "Reading
  metrics in your app" section shows a one-line integration with a
  third-party Prometheus exporter crate that `kremory` does not, and should
  not, depend on. Adding it to this harness purely to make that one line
  compile would be adding a dependency to make a test pass, not verifying a
  real capability — left as an honest compile failure.
- **README's "full trait" reference block** (`docs/api.md` has no equivalent)
  — README shows `pub trait EmbeddingProvider { ... }` as reference text
  right after a snippet that already `use`s the real `kremory::
  EmbeddingProvider`. Concatenated under this harness's `##`-grouping
  decision, the redeclaration collides with the import. Read standalone (as
  a human reads it) there is no conflict — this is flagged in the audit
  table as a harness-attributable finding, not a code-vs-doc defect.

## Output

Every run reports, per doc file, how many Rust blocks were extracted, how
many compilation units they were grouped into, and (via `cargo test --doc`'s
own output) which units failed with which compiler diagnostics — the direct
input to Phase 1's semantic audit table.
