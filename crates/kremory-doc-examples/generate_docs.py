#!/usr/bin/env python3
"""Phase 0 doc-compile extraction + generation harness.

docs/specs/public-docs-and-api-surface-audit/{spec,plan,tasks}.md — RULE-001,
RULE-002, T0.1-T0.4.

What this does
--------------
1. Walks docs/*.md + README.md.
2. Extracts every fenced ```rust code block, grouping consecutive blocks that
   sit under the same ``##``/``###`` heading into ONE compilation unit
   (RULE-001's "scaffold contract" — most blocks are narrative fragments that
   only make sense concatenated with their siblings, not standalone
   programs).
3. Detects non-compiling markers: a `no_run`/`compile_fail`/`ignore` token in
   the fence's info string, or the words "illustrative"/"pseudo-code" in the
   prose since the enclosing heading.
4. Wraps each compilation unit in a documented placeholder PRELUDE (RULE-001's
   minimum: `my_llm`, `my_embedder`, `MySink` — extended here with every other
   placeholder name discovered empirically against the real docs; see
   README.md in this directory for the full list + rationale).
5. Emits one generated markdown file per source doc into ./generated/ (NOT
   committed — see .gitignore), each containing one fenced block per
   compilation unit.
6. Asserts the total extracted-block count is >= MIN_TOTAL_RUST_BLOCKS
   (RULE-002's vacuous-pass guard) and fails loudly (non-zero exit) if not.

The generated files are consumed by `cargo test --doc -p kremory-doc-examples`
(see src/lib.rs in this crate, which wires one module per generated file via
`#[doc = include_str!(...)]` — this is option (b) from plan.md, SPIKED
2026-09-07 and confirmed to resolve `kremory::` imports and rustdoc's
no_run/compile_fail semantics correctly against a real fixture BEFORE this
full harness was built, per this project's own
mechanical-compile-spike-beats-paper-review rule).

Run:  python3 crates/kremory-doc-examples/generate_docs.py
Then: cargo test --doc -p kremory-doc-examples
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CRATE_DIR = Path(__file__).resolve().parent
OUT_DIR = CRATE_DIR / "generated"

# Every markdown file in scope for RULE-001 (the published doc corpus +
# README.md). Adding a new doc file requires adding it here AND wiring a
# matching `mod` in src/lib.rs.
#
# Both of those are places you can silently forget, which is why
# `check_doc_files_complete()` below asserts this list against the real tree
# rather than trusting it. RULE-002's `>= 60`-block floor is NOT that check:
# it catches a broken extractor, but a single dropped file whose fences leave
# the remaining count above 60 sails straight past it.
DOC_FILES = [
    REPO_ROOT / "docs" / "api" / "index.md",
    REPO_ROOT / "docs" / "api" / "setup.md",
    REPO_ROOT / "docs" / "api" / "namespaces.md",
    REPO_ROOT / "docs" / "api" / "ingest.md",
    REPO_ROOT / "docs" / "api" / "recall.md",
    REPO_ROOT / "docs" / "api" / "bi-temporal.md",
    REPO_ROOT / "docs" / "api" / "dream.md",
    REPO_ROOT / "docs" / "api" / "reversibility.md",
    REPO_ROOT / "docs" / "api" / "async-and-events.md",
    REPO_ROOT / "docs" / "api" / "advanced.md",
    REPO_ROOT / "docs" / "api" / "feature-flags.md",
    REPO_ROOT / "docs" / "api" / "node-binding.md",
    REPO_ROOT / "docs" / "releases" / "upgrade-guide.md",
    # NOTE: getting-started.md was absent from this list until 2026-09-08 —
    # its worked example had never been compiled. Found by the completeness
    # check below on its first run.
    REPO_ROOT / "docs" / "getting-started.md",
    REPO_ROOT / "docs" / "benchmarks.md",
    REPO_ROOT / "docs" / "comparison.md",
    REPO_ROOT / "docs" / "error-handling-policy.md",
    REPO_ROOT / "docs" / "eval-fixtures.md",
    REPO_ROOT / "docs" / "eval.md",
    REPO_ROOT / "docs" / "observability.md",
    REPO_ROOT / "docs" / "testing.md",
    REPO_ROOT / "README.md",
]

# Directories under docs/ that are NOT part of the published corpus and are
# therefore deliberately absent from DOC_FILES. Each needs a reason; anything
# not listed here and not in DOC_FILES makes the completeness check fail.
DOC_DIRS_EXCLUDED = {
    "specs": "process artefacts for the docs audit — not published to consumers",
    "adr": "architecture decision records — internal rationale, not consumer docs",
    "decisions": "decision records — internal, not consumer docs",
    "research": "research artefacts — internal, not consumer docs",
}


def check_doc_files_complete() -> None:
    """Fail loudly when DOC_FILES has drifted from the real doc tree.

    A file dropped from DOC_FILES stops being compiled while the harness keeps
    reporting green, so absence has to be an error rather than something a
    reader is expected to notice. Checks BOTH places a file can go missing:
    this list, and the hand-wired `mod` block in src/lib.rs.
    """
    on_disk = set()
    for md in (REPO_ROOT / "docs").rglob("*.md"):
        top = md.relative_to(REPO_ROOT / "docs").parts[0]
        if top in DOC_DIRS_EXCLUDED:
            continue
        on_disk.add(md.resolve())
    on_disk.add((REPO_ROOT / "README.md").resolve())

    listed = {f.resolve() for f in DOC_FILES}

    def rel(paths):
        return sorted(str(x.relative_to(REPO_ROOT)) for x in paths)

    missing, stale = on_disk - listed, listed - on_disk
    problems = []
    if missing:
        problems.append(
            "these doc files exist but are NOT in DOC_FILES, so their ```rust "
            f"blocks are not being compiled: {rel(missing)}"
        )
    if stale:
        problems.append(
            f"these DOC_FILES entries no longer exist on disk: {rel(stale)}"
        )

    # src/lib.rs wires one `mod` per generated file by hand — a second place to
    # forget, with the same silent-green failure mode.
    lib_rs = (CRATE_DIR / "src" / "lib.rs").read_text()
    for f in DOC_FILES:
        stem = f.relative_to(REPO_ROOT).as_posix().replace("/", "__")
        if f'generated/{stem}.generated.md' not in lib_rs:
            problems.append(
                f"{f.relative_to(REPO_ROOT)} is in DOC_FILES but has no "
                f"`#[doc = include_str!]` module in src/lib.rs, so it will not "
                f"be compiled"
            )

    if problems:
        raise SystemExit(
            "doc-compile harness is out of sync with the doc tree:\n  - "
            + "\n  - ".join(problems)
        )

# RULE-002: 60 ~= 87% of the 69-block count measured 2026-09-06/07. Vacuous-
# pass guard — if the extractor breaks (or the doc surface shrinks a lot),
# fail loudly rather than silently reporting "0 of 0 compile, 100% pass".
MIN_TOTAL_RUST_BLOCKS = 60

HEADING_RE = re.compile(r"^(#{2,3})\s+(.*)$")
FENCE_OPEN_RE = re.compile(r"^```(\S*)\s*$")
FENCE_CLOSE_RE = re.compile(r"^```\s*$")
ILLUSTRATIVE_RE = re.compile(r"illustrative|pseudo-?code", re.IGNORECASE)


@dataclass
class Block:
    file: Path
    group_id: int
    heading: str
    start_line: int  # 1-indexed line of the opening ``` fence
    end_line: int  # 1-indexed line of the closing ``` fence
    code: str
    kind: str  # "normal" | "compile_fail" | "excluded"
    exclude_reason: str = ""


@dataclass
class Unit:
    file: Path
    heading: str
    blocks: list[Block] = field(default_factory=list)
    unit_kind: str = "normal"  # "normal" | "compile_fail"

    @property
    def start_line(self) -> int:
        return self.blocks[0].start_line

    @property
    def end_line(self) -> int:
        return self.blocks[-1].end_line

    @property
    def line_range(self) -> str:
        if self.start_line == self.blocks[0].start_line and len(self.blocks) == 1:
            return f"{self.blocks[0].start_line}-{self.blocks[0].end_line}"
        return f"{self.start_line}-{self.end_line}"


def extract_file(path: Path) -> list[Block]:
    """Group boundary is `##` ONLY -- NOT `###` (design decision, see README.md
    "Grouping granularity"). RULE-001's literal text ("under one ##/### heading")
    reads as splitting on either level, but a real cross-###-subsection
    dependency exists in docs/api.md Section 6a ("SEE" binds `history`; "UNDO",
    a SEPARATE ### two headings later under the same ##, reads `history.first()`
    with no local definition) -- splitting there produces a FALSE "cannot find
    value `history`" failure that is an artifact of over-fine grouping, not a
    real doc defect. `###` text is still tracked and used for the heading LABEL
    (readability in the generated artifact + T0.4 reporting), just not as a
    fresh compilation-unit boundary.
    """
    lines = path.read_text().splitlines()
    blocks: list[Block] = []
    heading_text = "(preamble)"
    group_id = 0
    prose_since_heading: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        hm = HEADING_RE.match(line)
        if hm:
            heading_text = hm.group(2).strip()
            if hm.group(1) == "##":
                group_id += 1
            prose_since_heading = []
            i += 1
            continue
        fm = FENCE_OPEN_RE.match(line)
        if fm:
            info = fm.group(1)
            fence_start_line = i + 1  # 1-indexed
            body_lines: list[str] = []
            i += 1
            while i < n and not FENCE_CLOSE_RE.match(lines[i]):
                body_lines.append(lines[i])
                i += 1
            fence_end_line = i + 1  # 1-indexed line of the closing fence
            if i < n:
                i += 1  # consume the closing fence line

            tokens = [t.strip().lower() for t in info.split(",") if t.strip()]
            lang = tokens[0] if tokens else ""
            if lang == "rust":
                markers = set(tokens[1:])
                prose_text = "\n".join(prose_since_heading)
                if "compile_fail" in markers:
                    kind, reason = "compile_fail", "fence marked compile_fail"
                elif "ignore" in markers:
                    kind, reason = "excluded", "fence marked ignore"
                else:
                    im = ILLUSTRATIVE_RE.search(prose_text)
                    if im:
                        kind, reason = "excluded", f'prose marker: "{im.group(0)}"'
                    else:
                        # no_run and fully-unmarked are BOTH required to
                        # compile in Phase 0 (RULE-001: "compile != run" —
                        # Phase 0 never executes anything regardless of
                        # marker, see the forced no_run in render_unit()).
                        kind, reason = "normal", ""
                blocks.append(
                    Block(
                        file=path,
                        group_id=group_id,
                        heading=heading_text,
                        start_line=fence_start_line,
                        end_line=fence_end_line,
                        code="\n".join(body_lines),
                        kind=kind,
                        exclude_reason=reason,
                    )
                )
            prose_since_heading = []
            continue
        prose_since_heading.append(line)
        i += 1
    return blocks


def build_units(blocks: list[Block]) -> tuple[list[Unit], list[Block]]:
    """Group blocks per (file, heading-group) into compilation units.

    Design decision (documented in README.md): grouping keys on the nearest
    enclosing heading of EITHER level (## or ###) — i.e. every heading
    transition starts a new group, whichever level it is. Within a group,
    consecutive "normal" blocks are concatenated into ONE unit (narrative
    continuity — e.g. docs/api.md Section 1's `mem2 = mem.clone()` block
    needs `mem` from the preceding block in the same section). "compile_fail"
    blocks are extracted as their OWN isolated unit so a deliberately-broken
    example never corrupts the pass/fail signal of its siblings. "excluded"
    blocks produce no unit at all.
    """
    order: list[tuple[Path, int]] = []
    groups: dict[tuple[Path, int], list[Block]] = {}
    for b in blocks:
        key = (b.file, b.group_id)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(b)

    units: list[Unit] = []
    excluded: list[Block] = []
    for key in order:
        gblocks = groups[key]
        mergeable = [b for b in gblocks if b.kind == "normal"]
        isolated = [b for b in gblocks if b.kind == "compile_fail"]
        excluded.extend(b for b in gblocks if b.kind == "excluded")
        if mergeable:
            units.append(Unit(file=key[0], heading=mergeable[0].heading, blocks=mergeable, unit_kind="normal"))
        for b in isolated:
            units.append(Unit(file=key[0], heading=b.heading, blocks=[b], unit_kind="compile_fail"))
    return units, excluded


# ---------------------------------------------------------------------------
# Prelude — see README.md "Prelude contents" for the full rationale per name.
# ---------------------------------------------------------------------------

PRELUDE_HEADER = """\
#![allow(unused, dead_code, unused_variables, unused_imports, unused_mut)]
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use chrono::{Utc, Duration as ChronoDuration, DateTime, TimeZone};
use kremory::{Memory, Namespace, DynEmbeddingProvider, EnrichmentEventSink, IngestEventSink, ChatProvider};
use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
// NOTE: `EmbeddingProvider` is deliberately NOT glob-imported here. One real
// doc block (README.md "The full trait (`kremory::EmbeddingProvider`):")
// re-shows the trait's own definition as reference text -- `pub trait
// EmbeddingProvider { ... }` -- which collides with a blanket `use` of the
// same name once merged into the same compilation unit as sibling blocks.
// Referenced by full path (`kremory::EmbeddingProvider`) below instead; any
// doc block that needs the trait in scope for method-call syntax already
// brings it in with its own local `use`, matching how it reads standalone.

// --- kremory doc-example prelude (RULE-001 scaffold contract) ---------------
// See crates/kremory-doc-examples/README.md for the full rationale. `MySink`
// is the minimum placeholder RULE-001 names explicitly; every other name here
// was discovered empirically against the real docs during T0.1/T0.2.

/// Minimum RULE-001 placeholder: a unit struct implementing EnrichmentEventSink.
struct MySink;
impl IngestEventSink for MySink {}
impl EnrichmentEventSink for MySink {
    fn on_community_updated(&self, _id: &str, _count: usize) {}
    fn on_batch_phase2_complete(&self, _e: kremory::BatchPhase2Complete) {}
}

/// docs/observability.md names a reader-defined chat provider
/// `MyCustomChatProvider::new()` without ever showing its body -- this is a
/// stand-in with a real (delegating) ChatProvider impl so the surrounding
/// `.with_llm(Arc::new(MyCustomChatProvider::new()))` call type-checks.
#[derive(Clone)]
struct MyCustomChatProvider(MockChatProvider);
impl MyCustomChatProvider {
    fn new() -> Self {
        Self(MockChatProvider::null())
    }
}
#[async_trait::async_trait]
impl ChatProvider for MyCustomChatProvider {
    async fn chat_with_tools(
        &self,
        messages: &[autoagents_llm::chat::ChatMessage],
        tools: Option<&[autoagents_llm::chat::Tool]>,
        json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
    ) -> std::result::Result<Box<dyn autoagents_llm::chat::ChatResponse>, autoagents_llm::error::LLMError>
    {
        self.0.chat_with_tools(messages, tools, json_schema).await
    }
}

/// Sibling of MyCustomChatProvider for the matching
/// `MyCustomEmbeddingProvider::new()` placeholder in the same doc.
#[derive(Clone)]
struct MyCustomEmbeddingProvider(NullEmbeddingProvider);
impl MyCustomEmbeddingProvider {
    fn new() -> Self {
        Self(NullEmbeddingProvider { dim: 384 })
    }
}
impl kremory::EmbeddingProvider for MyCustomEmbeddingProvider {
    fn embed<'a>(&'a self, text: &'a str) -> impl std::future::Future<Output = kremory::CoreResult<Vec<f32>>> + Send + 'a {
        // `as _`: bring the trait into scope for method-call resolution ONLY
        // (no name bound), so it can never collide with a doc block that
        // re-declares `EmbeddingProvider` itself as reference text elsewhere
        // in this same compilation unit.
        use kremory::EmbeddingProvider as _;
        self.0.embed(text)
    }
}
"""

# Emitted ONCE, before the first original block.
PRELUDE_BODY_ONCE = """\
    // --- kremory doc-example prelude bindings (RULE-001) ------------------
    // Persistent seeds. PRELUDE_REFRESH_LINES (below) is re-emitted VERBATIM
    // before every concatenated original block so that reusing the same
    // placeholder name (my_llm, llm, l, ...) across multiple
    // ORIGINALLY-INDEPENDENT snippets never trips a "value moved" error that
    // would be an artifact of concatenation, not a real doc-vs-code defect.
    // NOTE: this is plain textual repetition, not a `macro_rules!` -- Rust
    // macro hygiene makes `let` bindings created inside a macro invisible to
    // code outside that specific invocation, which broke this exact idea on
    // the first real run (T0.2 spike finding, 2026-09-07).
    let __llm_seed: MockChatProvider = MockChatProvider::null();
    let __emb_seed: NullEmbeddingProvider = NullEmbeddingProvider { dim: 384 };
    // Most sections after Section 1 (Quickstart) assume a `mem` already
    // exists -- they never construct one locally. Build one real, working
    // `Memory` up front so every later per-section unit has it in scope.
    let mem: Memory = Memory::open("/tmp/kremory-doc-examples-generated.db")
        .with_llm(Arc::new(__llm_seed.clone()))
        .with_embedder(Arc::new(__emb_seed.clone()))
        .default_namespace(Namespace::new("doc-example-namespace"))
        .await?;
"""

# Re-emitted verbatim before EVERY original block (see PRELUDE_BODY_ONCE note).
PRELUDE_REFRESH_LINES = """\
    #[allow(unused)]
    let my_llm = __llm_seed.clone();
    #[allow(unused)]
    let my_embedder = __emb_seed.clone();
    // `llm` / `emb` (unlike `my_llm` / `my_embedder`) are pre-wrapped in `Arc<dyn ...>`:
    // most doc sections that use these short names call `.with_llm(llm)` /
    // `.with_embedder(emb)` bare, matching the convention of "you already have an
    // Arc'd provider in scope" (as a real consumer following the doc top-to-bottom
    // would, from an earlier section) -- vs `my_llm` / `my_embedder`, which several
    // OTHER sections wrap explicitly via `Arc::new(my_llm)`. Discovered empirically
    // (2026-09-07): the un-wrapped form was a harness bug relative to its own stated
    // intent -- see this file's README "llm, emb, l, e" prelude row.
    #[allow(unused)]
    let llm: Arc<dyn ChatProvider> = Arc::new(__llm_seed.clone());
    #[allow(unused)]
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(__emb_seed.clone());
    #[allow(unused)]
    let l = __llm_seed.clone();
    #[allow(unused)]
    let e = __emb_seed.clone();
    #[allow(unused)]
    let ns = Namespace::new("doc-example-namespace");
    #[allow(unused)]
    let x = Utc::now();
    #[allow(unused)]
    let y = Utc::now();
    // Discovered empirically (T0.2, not in RULE-001's starter list):
    #[allow(unused)]
    let turn_text = "a conversation turn";
    #[allow(unused)]
    let question = "a recall query";
    #[allow(unused)]
    let fact_id: i64 = 1;
"""


# Names the prelude header ALREADY brings into scope at module level. A doc
# block's own local `use` of any of these is redundant on its own (legal --
# shadowing an outer name from inside a fn body is fine) but becomes E0252
# ("name defined multiple times") the moment a SECOND original block, now
# concatenated into the SAME function scope, imports the same name again.
# This is a harness artifact of merging independently-written fragments, not
# a real doc defect -- deduped below rather than reported as a finding.
PRELUDE_IMPORTED_NAMES = {
    "Memory", "Namespace", "DynEmbeddingProvider", "EnrichmentEventSink",
    "IngestEventSink", "ChatProvider", "MockChatProvider", "NullEmbeddingProvider",
}

_USE_LIST_RE = re.compile(r"^(\s*)use\s+([\w:]+)::\{([^}]*)\}\s*;\s*$")
_USE_SINGLE_RE = re.compile(r"^(\s*)use\s+([\w:]+)::(\w+)\s*;\s*$")


def dedupe_block_code(code: str, seen: set[str]) -> str:
    """Elide a name from a `use` statement if already in scope (see
    PRELUDE_IMPORTED_NAMES doc above). Handles both single-line
    `use path::{A, B};` and multi-line variants (joins continuation lines up
    to the terminating `;`). Anything it can't confidently parse (globs, `as`
    aliases -- neither appears in this doc corpus, checked by grep before
    writing this) is passed through unchanged.
    """
    lines = code.splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("use ") and "::" in stripped:
            buf = [line]
            j = i
            while ";" not in lines[j] and j + 1 < n:
                j += 1
                buf.append(lines[j])
            oneline = " ".join(x.strip() for x in buf)
            m = _USE_LIST_RE.match(oneline)
            m2 = _USE_SINGLE_RE.match(oneline) if not m else None
            indent = re.match(r"^(\s*)", line).group(1)
            if m:
                _, path, names_str = m.groups()
                names = [nm.strip() for nm in names_str.split(",") if nm.strip()]
                keep = [nm for nm in names if nm not in seen]
                seen.update(names)
                if keep:
                    out.append(f"{indent}use {path}::{{{', '.join(keep)}}};")
                else:
                    out.append(f"{indent}// (use elided -- already in scope: {', '.join(names)})")
                i = j + 1
                continue
            if m2:
                _, path, name = m2.groups()
                if name in seen:
                    out.append(f"{indent}// (use elided -- already in scope: {name})")
                else:
                    seen.add(name)
                    out.append(f"{indent}use {path}::{name};")
                i = j + 1
                continue
            out.extend(buf)
            i = j + 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def render_unit_fence(unit: Unit) -> str:
    marker = "compile_fail" if unit.unit_kind == "compile_fail" else "no_run"
    # Phase 0 is a type-check ONLY (RULE-001): every "normal" unit is forced
    # no_run regardless of whatever marker the original block(s) carried, so
    # nothing in this harness ever makes a network call or touches a real
    # filesystem path beyond compiling.
    parts = [PRELUDE_HEADER, "", "#[tokio::main]", "async fn main() -> anyhow::Result<()> {", PRELUDE_BODY_ONCE]
    seen_use_names: set[str] = set(PRELUDE_IMPORTED_NAMES)
    for b in unit.blocks:
        rel = b.file.relative_to(REPO_ROOT)
        parts.append(f"    // --- {rel}:{b.start_line}-{b.end_line} (heading: {b.heading!r}) ---")
        parts.append(PRELUDE_REFRESH_LINES.rstrip("\n"))
        deduped_code = dedupe_block_code(b.code, seen_use_names)
        for line in deduped_code.splitlines():
            parts.append("    " + line if line.strip() else "")
    parts.append("    Ok(())")
    parts.append("}")
    body = "\n".join(parts)
    rel = unit.file.relative_to(REPO_ROOT)
    title = f"#### {rel} — {unit.heading} (lines {unit.line_range})"
    return f"{title}\n\n```rust,{marker}\n{body}\n```\n"


def generate() -> tuple[list[Unit], list[Block], int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_blocks: list[Block] = []
    per_file_units: dict[Path, list[Unit]] = {}
    for doc in DOC_FILES:
        blocks = extract_file(doc)
        all_blocks.extend(blocks)
        units, _ = build_units(blocks)
        per_file_units[doc] = units

    for doc, units in per_file_units.items():
        out_name = doc.relative_to(REPO_ROOT).as_posix().replace("/", "__") + ".generated.md"
        out_path = OUT_DIR / out_name
        header = f"<!-- GENERATED by generate_docs.py from {doc.relative_to(REPO_ROOT)} — DO NOT EDIT -->\n\n"
        content = header + "\n".join(render_unit_fence(u) for u in units)
        out_path.write_text(content)

    rust_blocks = [b for b in all_blocks if True]  # already rust-only (extract_file only appends rust)
    return (
        [u for units in per_file_units.values() for u in units],
        rust_blocks,
        sum(1 for b in rust_blocks if b.kind == "excluded"),
    )


def main() -> int:
    # Before anything else: a file missing from DOC_FILES silently stops being
    # compiled, and every downstream number would still look healthy.
    check_doc_files_complete()
    units, rust_blocks, excluded_count = generate()
    total = len(rust_blocks)
    print(f"Extracted {total} rust code blocks across {len(DOC_FILES)} doc files.")
    print(f"  -> {len(units)} compilation units generated into {OUT_DIR}")
    print(f"  -> {excluded_count} blocks excluded (marker/prose opt-out)")

    per_file_counts: dict[str, int] = {}
    for b in rust_blocks:
        rel = b.file.relative_to(REPO_ROOT).as_posix()
        per_file_counts[rel] = per_file_counts.get(rel, 0) + 1
    for rel, count in sorted(per_file_counts.items()):
        print(f"    {rel}: {count}")

    if total < MIN_TOTAL_RUST_BLOCKS:
        print(
            f"\nFATAL (RULE-002 vacuous-pass guard): extracted only {total} rust "
            f"blocks, need >= {MIN_TOTAL_RUST_BLOCKS}. Extractor is broken or the "
            "doc surface shrank a lot -- refusing to report a pass/fail number "
            "against a suspiciously small extraction.",
            file=sys.stderr,
        )
        return 1

    print(f"\nRULE-002 guard: {total} >= {MIN_TOTAL_RUST_BLOCKS} -- OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
