<!-- sprint-activate-begin -->
## ⚠️ READ FIRST — System Primer (every session, before investigating/claiming anything)

**`.ai-docs/SYSTEM-PRIMER.md`** is the canonical, verified system model + gotchas + bench runbook +
current recall/known-issue state. Read it before benchmarking, debugging, or asserting anything about
kremory. It exists so we STOP re-deriving the same things. #1 gotcha: **`COUNT(*)` returns 0 on the
`entities`/`facts` vector-indexed tables even when populated** — count via `SELECT recorded_at FROM
entities | wc -l`. The graph on `.context/full-corpus.db` is populated (1197 entities, 5148 facts).
Correct the primer in place when reality changes (verify vs code, not memory).

---

## Active Sprint

## ⚠️ CURRENT STATE — verified against code + a real gate run, 2026-08-06

**This supersedes the 0.3.2-era block below, which is retained for provenance only. Where the
two disagree, THIS wins.**

| fact | value |
|---|---|
| tree version | ~~`Cargo.toml` says **0.6.0**~~ — **stale, `Cargo.toml` now says 0.7.0 (bumped 2026-09-02).** 0.6.0 was the last version live on crates.io as of session start (below). Many commits touch `crates/` since that publish, incl. TD-206/TD-203/TD-231/TD-232 fixes and the ADR-068 as_of-for-content-search extension. **`kremory-napi` is NOT the same gate** — corrected 2026-09-02: it has never been published to crates.io (`curl .../crates/kremory-napi` → `"does not exist"`), and now never will be — `publish = false` set in its `Cargo.toml` deliberately (2026-09-02), because it's a workspace-internal napi-rs build artifact with zero real Rust consumers; its actual (unstarted) distribution surface is npm, `@kgentic-ai/kremory-node`, a wholly separate registry/publish step. ~~Its version field still tracks alongside `kremory`'s for internal consistency~~ **— WRONG, corrected 2026-09-05: it does not. `kremory-napi` is at `0.4.1` (Cargo.toml AND package.json agree) while `kremory` is at `0.7.1`. The two have not tracked each other for three minor versions.** That number is not a publish-readiness signal for it either way. ~~Tree version is now `0.7.1` (bumped 2026-09-05), PREPARED BUT NOT PUBLISHED — crates.io still holds 0.7.0.~~ **Bumped again to `0.8.0` on 2026-09-08: two BREAKING API changes landed on top of the unpublished 0.7.1 prep (`DreamRequest` now requires `.execute()`; `NoEmb`/`WithEmb` renamed to `NoEmbedder`/`WithEmbedder` — see `crates/kremory/CHANGELOG.md` `[0.8.0]`). Per semver for a pre-1.0 crate, a breaking change bumps the minor (0.7.x → 0.8.0), not the patch. Nothing between 0.7.0 and 0.8.0 was ever published, so 0.7.1 never existed as a live version — 0.8.0 absorbs everything prepared since 0.7.0. `cargo publish` remains a standing HITL gate, not exercised here.** |
| **crates.io** | ✅ **0.6.0, live since 2026-08-11T11:59:52Z, `default = ["content-search"]`.** `cargo add kremory` gets the CORRECT default. ~~**0.5.0**, `default = []` — every user gets search OFF (~25.7% vs ~86.2%)~~ **← WRONG-VERIFIED, corrected 2026-08-12.** That claim was false for >24h and was inherited by the 2026-08-12 session note and the consolidation plan, whose flagship "Phase 0 — publish the fix" proposed re-doing work already done. Verified two independent ways: `curl https://index.crates.io/kr/em/kremory` (sparse index shows `"default":["content-search"]` on 0.6.0) and `curl https://crates.io/api/v1/crates/kremory` (`max_version: 0.6.0`). **Re-check the registry, not the doc, before ever asserting what users receive.** |
| `default` features | `["content-search"]` in **all three** crates (`kremory`, `kremory-mcp`, `kremory-napi`) — verified 2026-08-06 |
| workspace nextest | **STATE THE SCOPE — two different numbers, both correct.** `-p kremory -p kremory-mcp` (`content-search,test-utils`) → ~~1699~~ ~~1742~~ ~~1748~~ **1790 passed / 1790 run, 6 skipped, re-measured 2026-09-06 (end of session)** at scope `-p kremory -p kremory-mcp --features content-search,test-utils`. Grew 1751 -> 1772 by exactly the 21 tests added that day: 11 for ADR-080 prior-turn replay (7 graph-layer + 4 prompt-render), 6 for the `update_episode_metadata` multi-row clobber fix, 4 for the dream-reclassify registry bounds check. Every one was verified RED before being trusted — by neutering the mechanism it guards and confirming the failure, not by assumption. Then +13 in the same day's debt-clearance pass: 10 for TD-235 namespace-scoping the episode metadata/uri writers, 3 for the ADR-reference-integrity guard and the consistency-check registry bounds check. Prior figures this session: ~~1772~~ ~~1751 passed / 1751 run, 6 skipped, 2026-09-05~~ (+3 = the `public_surface_hygiene` guard, its own sensitivity test, and an exemption-liveness test; both verified RED-on-injected-leak before being trusted). Prior figure: (grew +49 across this session: TD-231/TD-232 tests, 10 ADR-068 as_of-for-content-search tests, 6 TD-232/TD-234 `split_for_embedding` tests — 4 direct unit tests on the splitter itself + 2 real end-to-end tests in `tests/it/td234_split_for_embedding_closes_the_loop.rs` reproducing TD-232's own oversized-document fixture; all new tests this session verified sensitive via neuter-and-confirm-RED before being trusted). Workspace-wide → **1782/1782, 6 skipped** (2026-08-06, not re-measured at workspace scope this session — the `-p kremory -p kremory-mcp` number above is the one actually re-verified). The 2026-08-12 run grew **1682 → 1699** by exactly the 17 tests added that day: 4 lexical date-merge, 1 zero-padding regression, 4 write-gate temporal veto, 8 NULL-embedding gap-fill. ⚠️ An earlier session read the 1682-vs-1782 gap as a REGRESSION and nearly halted on it; it is a scope difference. Never write a test count here without its package scope AND feature flags. |
| doctests | `-p kremory` → ~~25~~ ~~26~~ ~~27~~ **26 passed, 5 ignored, re-verified 2026-09-04** (direct `cargo test -p kremory --doc` run, pre-release gate for 0.7.0 — the prior "27" figure from 2026-09-02 does not reproduce; re-measured, not assumed, per this doc's own verify-before-stating discipline). `--workspace` → **25 passed, 6 ignored** (2026-08-06, not re-measured workspace-wide this session). ⚠️ nextest does NOT run these — separate tier. **State the scope or the number is meaningless**: the 6th ignored is `kremory-eval/src/layer_b/graph_integrity.rs:19`; `kremory-mcp`/`kremory-napi` have 0 and `kremory-admin` has no lib target at all. A prior "CLAUDE.md says 5, measured 6" drift note was NOT a drift — both numbers were right at different scopes. |
| clippy | `--workspace --all-targets --all-features` → **exit 0** |
| consumer E2E | **5 passed / 0 failed of 5** (real Ollama). Extended 2026-08-06 with `.content()` / `.as_of()` / `supersede`-reachability — all three ran in **all 5** runs, and `as_of(−10y) → 0 facts` vs `as_of(+1d) → 10–15` held **every** run |
| parity skip-list | **90 / cap 90 — AT the cap, raised deliberately 2026-09-06 (ADR-080).** Measured, not read: `grep -c '^\[\[skip\]\]' crates/kremory-napi/parity-skip.toml` = 90, and the enforced assert is `skip_count <= 90` at `api_parity.rs:494`. Raised 89 -> 90 for ONE entry (`MemoryBuilder::prior_turn_replay_depth`, the eleventh per-knob override setter, same ADR-030 Form B deferral as its ten siblings). ⚠️ **When raising this cap, grep the gate file for the OLD number — it is written in THREE places (lead comment, the assert, and a 'now both N' note) and I missed the third on the first pass, which is exactly how this row rotted before.** Historical, kept for provenance: ~~88 / cap 88~~ ~~89 / cap 89~~ (`parity-skip.toml` holds exactly 89 `[[skip]]` entries; the enforced assert is `skip_count <= 89` at `api_parity.rs:493`, raised 88 -> 89 on 2026-09-02 for TD-231). **This is the FOURTH time this row has been wrong** — the gate file's own comment says the note there "ROTTED TWICE", and this doc then rotted again behind it. The operational conclusion has never changed and is the only part worth trusting: it is AT the cap, so the next added skip fails the gate. Historical: **88 / cap 88 — AT the cap**; the next added skip fails the gate. ~~86 / cap 86~~ corrected 2026-08-12: the enforced assert is `skip_count <= 88` (`api_parity.rs:484`) and `parity-skip.toml` holds exactly 88 `[[skip]]` entries. The operational conclusion is unchanged — it is AT the cap — only the numbers were wrong, for the **third** time on this row. |

**🛑 Standing HITL gates — no autonomous action on any of these:** `cargo publish` · `git push` ·
release tags · **any paid benchmark run, which INCLUDES re-ingesting the bench corpora** (that
path uses the paid Groq extraction model, not local Ollama).

**⚠️ THE TOOL SHELL IS `zsh`, NOT fish — corrected 2026-08-12.** Every doc in this repo (and
three reviewer prompts written today) says *"the shell is fish"*. Your LOGIN shell is fish, but
**agent tool-calls execute under `/bin/zsh` 5.9** (`ZSH_VERSION=5.9`, `BASH_VERSION` and
`FISH_VERSION` both unset). Consequences, all measured:
- **The pipe hazard is REAL here too:** `false | tail -1; echo $?` → **0**. Keep capturing exit
  codes off the binary (`cmd > file 2>&1; echo $?`), never through a pipe.
- **`echo $status` DOES work** — zsh defines `status` as a synonym for `?`. The documented
  remedy is fine; it works for a different reason than the docs claim. (This was nearly written
  up as "the documented fix is broken" — it isn't. Tested first.)
- **An unmatched glob ABORTS THE WHOLE COMMAND.** `(eval):1: no matches found: <pat>` is zsh's
  `nomatch`, not fish. This bit three times today: `grep -rn X crates/ --include=*.rs` died
  before running because `*.rs` matched nothing in the CWD, so a "no matches" conclusion was
  really "the command never ran". **QUOTE the pattern**: `'--include=*.rs'`.

**⚠️ macOS Gatekeeper tax on a cold full relink — LARGELY RESOLVED 2026-08-06.** After a fresh
`--all-features` build, nextest appears to hang before printing `Starting N tests`. It is not
hung: each freshly-linked test binary must be executed once for `syspolicyd` to clear it, at
**~34s each**. **This used to mean ~180 binaries ≈ 1h** — measured again on 2026-08-06 at
**~35s/binary with `syspolicyd` pegged at ~65%, taking 90 of that run's 120 minutes.**

Consolidating `crates/kremory/tests/*.rs` (157 separate test binaries) into the single `it`
binary cut the workspace from **180 → 24 binaries**, so the tax is now seconds, not an hour.
The remaining 24 can still stall briefly on a cold relink; if they do, warm them **serially**
(`<binary> --list` in a loop) for visible progress. **Scope the warm to `target/debug/deps/`
only** — the naive binary list also contains real `src/bin` targets, one of which
(`entity_extraction_baseline`) rewrites a COMMITTED baseline file when executed.

**Consequence for `cargo test --test <name>`:** kremory has exactly ONE integration-test target
now. `--test foo` no longer resolves; use `--test it foo::` or, preferably,
`cargo nextest run -p kremory -E 'test(/^foo::/)'`.

**Live plan:** `.ai-docs/plans/v1-debt-and-gap-closure-2026-08-06.md`. Ordering note: the
benchmark is a **GO/NO-GO gate ON publishing**, not a follow-up to it.

---

~~**No active sprint.** Last shipped: **kremory 0.3.2** (live on crates.io — episodic-edge presence-uniqueness fix). `origin/main` = `fad4d5d` (0.3.2 + TD-053 napi-parity hygiene). The post-v0.2.4 Tech-Debt Clearance sprint and the 0.3.x release line are **DONE and merged to main** — the prior "30 commits ahead / UNPUSHED / v0.2.4 bundle" framing is retired (it predated the 0.3.x releases).~~ **← SUPERSEDED, see the table above.**

**Current baseline (~~verified 2026-06-25~~ — historical, superseded above):**

- Releases **0.3.0 → 0.3.1 → 0.3.2** shipped, tagged, on crates.io; `main` fast-forwarded through all. 0.3.0 yanked (broken default); 0.3.1 + 0.3.2 live.
- **TD-053 (napi parity hygiene) CLOSED + on main** (`a05be15` doc-rot pass, `fad4d5d` boy-scout pass): ~~`parity-skip.toml` = **72 entries ≤ 80 cap**~~ ~~**← corrected 2026-08-03: cap 83, sitting 83/83**~~ **← THAT correction is ALSO stale. Re-measured against the code 2026-08-06: the cap is 86 (`api_parity.rs:474`) and `parity-skip.toml` holds exactly 86 `[[skip]]` entries — 86/86.** The 83 figure missed two later raises, both on 2026-08-03 (83→84 for `with_provider_rates_path`; 84→86 for the two symbols the REPAIRED PAR-G1/G2 walker surfaced on its first run). `cargo test -p kremory-napi --test api_parity` **7/0 GREEN**. ⚠️ **The operational conclusion was right and is unchanged: it is AT the cap, so the next added skip fails the gate — treat 86/86 as a live constraint, not headroom.** Only the numbers were wrong, which is why this was worth re-measuring rather than re-reading. (`api_parity.rs:435`'s own lead comment still says "must not exceed 83" while `:474` asserts 86 — the same doc-rot, inside the gate file itself.) The napi 1:1 surface is verified 1:1 (ADR-034 landed); parity-skip now lists only genuinely-absent symbols.
- **Standing test fails: NONE (TD-138 RESOLVED 2026-07-23).** The 3 recall-branch fails (`facade_as_of_warn` ×2 + `with_facts_integration::td116_recall_returns_connected_facts_under_null_embedder`) were a silent recall regression — `rrf_fuse_with_content`'s no-limit cap `entity_count.max(content_count)` (from TD-066 Increment 1) dropped a distinct arm's items, evicting the entity under `content-search`. Fixed: no-limit cap = full union (`entity_count + content_count`); explicit-limit path unchanged. Workspace nextest now ~~**1516/1516**~~ **1782/1782 (6 skipped), re-measured 2026-08-06** (`content-search,test-utils`). See TD-138. (TD-013 `with_facts_empty_vec_equivalent_to_no_facts` was verified **CLOSED 2026-07-15**.) The old "napi parity-cap fail" was never real (the "103>100" number was fabricated per `feedback_subagent_fabricates_gate_results`). NB: this "Active Sprint" block still describes the 0.3.2 line and is broadly stale — current baseline is **0.5.0** (crates.io); see `.ai-docs/planning/pre-public-launch-readiness-roadmap-2026-07-15.md` + the MVP-to-public-flip execution plan for live state.
- `cargo fmt --check` is NOT hard-enforced (main has drift in `sink_fires_through_ingest.rs`).
- The napi parity test is invisible to `cargo test -p kremory` (cross-crate gate) — run workspace-wide to see it.

**Standing constraints (load-bearing for ANY release / push):**

- **No CI** — `kgentic` org GitHub Actions billing suspended (no restore expected). All 4 workflows (`ci` / `napi-ci` / `release-please` / `eval-canary`) are FULLY COMMENTED OUT (RESTORE header + `git revert` on each). Cross-platform binary builds + crate/npm publishes are **manual** off the M4 Max.
- **Push + release-tag = irreversible HITL gate** — no autonomous push or tag; user confirms each. Clean fast-forward to `main` is the normal land path for hygiene commits.
- **gah convention** — `.gahrc` present (work hours 9–17). Run gah before push to shift unpushed commits out-of-hours; it no-ops when commits are already OOH, and its scope is the unpushed range only (`origin/main..HEAD`). gah refuses on a dirty tree — untracked files from a parallel agent count as dirty (skip gah when already-OOH in that case).

**Open backlog** — `.ai-docs/tech-debt/tech-debt-register.md` (living) is the SoT. Still open at last review: TD-013 (above); TD-043 / TD-045 god-file splits (`graph.rs`, `ingest/pipeline.rs` — PARTIAL, verify current LoC against the register before resuming); TD-005 / TD-012 / TD-016 / TD-029 / TD-030 / TD-032 / TD-034 (pre-existing deferrals); TD-H (RISK-001 multi-run mean±SD hardening); **TD-086 / TD-087 / TD-088 / TD-089 (dream-phase debt cluster). UPDATED 2026-07-01: the prior "live `mem.dream()` runs only 2 of 5 passes" headline is STALE — all 5 reconciliation passes (discover / aliases / reclassify / consistency_check / canonicalize) are now wired into live `mem.dream()` (TD-089 Phases 1-5 committed on `kgentic-dev/kremory-real-llm-suite`; verified `facade/dream.rs` call sites). The model-threading bug that ran the 3 LLM passes empty (TD-094) is RESOLVED 2026-07-01. Remaining dream write-side work: TD-089 Phase 6 cleanup (delete dead `run_dream_phase_passes` orchestrator, supersede 2026-06-05 spec) + dream-quality follow-ups (TD-095 Lane A discovery quality etc.). TD-086/087/088 current status: verify against the register — the parallel dream-v2 agent's active lane.** Verify any TD's status against the register + actual code before acting — register narrative can lag.

**Continuous-improvement dream direction (NEW 2026-07-01, deferred/OOS):** ADR-061 captures the "gets smarter while you sleep" direction (6 axes A–F, risk order C→B→A) backed by a 9-system prior-art swarm (`.ai-docs/research/dream-continuous-reconciliation/SYNTHESIS.md`). Axis-C read-time graph-proximity re-rank is RATIFIED (`/ship-architect` + Vera, 2 cycles) — decision **ADR-062**, build-entry spec `.ai-docs/specs/axis-c-read-time-relevance-spec-2026-07-01.md` (build-ready, spike-gated; prerequisite: a single-namespace-recall sort-order fix TD, not yet filed — register was mid-edit). Axis B/A remain at direction-capture in ADR-061.

**Graph query-language + schema-model teardown (NEW 2026-07-02, research-complete / ADR-064 proposed):** A maintainer question ("is our triple-table schema correct?") triggered a Cypher-over-libSQL feasibility study (`.ai-docs/research/cypher-over-libsql-2026-07-02.md`) + a 6-competitor graph-model teardown (`.ai-docs/research/graph-model-competitor-teardown-2026-07-02.md`, vs Neo4j / Kuzu / Apache AGE / DuckPGQ / CozoDB / SQLite-native). Outcome — **ADR-064** (proposed): the triple-table-over-libSQL model is SOUND (Graphiti-convergent; bi-temporal ahead 6-for-6) — KEEP it. Adoption backlog: **TD-099** (CSR traversal representation — the top "steal"), **TD-100** (`graph_edges` view for the edge/literal split), **TD-101** (predicate covering indexes), **TD-102** (drop vestigial `facts.access_count`). A Cypher-subset query language is feasible but UNSCHEDULED. Rejected: entity primary-key-id integer migration (the name-slug PK — NOT the already-integer `entity_type_id`, which shipped via ADR-056 and is untouched), AGE per-label tables, SQL/PGQ standard, adopting an external engine. Correction folded into the cypher doc: variable-length traversal stays in-memory (CSR/petgraph), NOT recursive-CTE — AGE/DuckPGQ/Kuzu all avoid SQL recursion for it.

**Deferred feature (captured, not scheduled):** **ADR-059** (proposed, OOS) — consumer-directed dream + relationship-discovery query (the read/write seam: free-text steering safe only on the read/recall surface; write-side directed dream takes structured scope only). Route through `/ship-architect` + Vera if/when it goes in-scope.

**In-flight adjacent work (other agent's lane — do NOT collide):**

- **ADR-053** (proposed) — TS/JS MCP server distribution shape (Hybrid A→B) + the companion architecture doc + ADR-054 cloud-positioning. Untracked in this workspace; a parallel agent owns them. The 0.3.2 fix + TD-053 hygiene are inherited by that release via the `kremory = { path = ... }` dependency. A handoff note (incl. a `recall({query})` doc-error flag for their arch doc §5) sits at `.context/NOTE-to-tsmcp-distribution-agent-2026-06-25.md`.

**Strategic posture (load-bearing for future planning):**

- `cloud-readiness-posture-2026-06-11` — passive cloud-readiness, no speculative engineering. Apply the 30-second self-check at every future ADR/spec.
- `queue-and-background-worker-research-2026-06-11` — recommended hybrid P1+P2 (tokio mpsc + apalis SQLite) when durability becomes a requirement.
- ADR-048 (accepted, **v0.3.0 scope**) — three-signal local-first consistency check.

**Recurring-discipline reminders (from prior sprints, still apply):**

- Global clippy threshold changes must gate **both** `--all-targets` AND `--all-features` — cfg-gated code (e.g. `ner.rs` under `--features ner`) is invisible to default-features clippy (`feedback_gate_all_features_for_global_clippy_changes`).
- Whole-program-coupled refactors (args-as-object splits) run **sequential + foreground only** — background agents zombie on machine-sleep and race the orchestrator (`feedback_background_agents_zombie_on_machine_sleep`).
- When other agents are modifying the tree, commit with **explicit paths** (`git commit -- <paths>`), never a bare commit that sweeps their staged/untracked work.

### Phase-boundary discipline (preserved across sprints)

Every phase commit MUST satisfy these gates in order — they apply to any sprint type unless explicitly waived in the sprint plan:

1. **`/quality-gate` PASS** — typecheck + lint + test + build per CLAUDE.md "Pre-Commit Gate NON-NEGOTIABLE". No band-aids — cause-fix any failure per Rule 8.
2. **Quinn `/ship-build-review` PASS or CONCERNS-resolved** — Sonnet adversarial review over the phase's aggregate diff. HIGH findings BLOCK; MEDIUM findings folded into the same PR or follow-up; LOW findings filed as new TDs OR rolled into boy-scout sweep per `feedback_boy_scout_includes_quinn_low_findings`.
3. **LLM integration smoke PASS** — phase-specific real-LLM end-to-end verification when relevant (skip for pure-docs sprints).
4. **CONSUMER E2E TIER — `./scripts/run-e2e-consumer.sh 3` — MANDATORY BEFORE ANY RELEASE.**
   Not per-commit (it needs Ollama and takes ~4 min/run), but **no release is cut without
   it**. It is the ONLY tier that drives the **published surface** as an external consumer
   against a real model; every other test builds from the tree and calls internals.

   **Why it is a hard release gate.** Nothing ran it for weeks, and in that window ADR-078's
   `content-search` default flip silently broke the consumer journey (`recall().raw()` began
   returning content passages in a field named `entity_id`). 1,772 deterministic tests stayed
   green throughout and could not see it. The harness's own lockfile was still pinned at
   kremory 0.4.0 — proof nobody had run it since before the 0.5.0 bump.

   **Run it ≥3× and read the RATE, never a single result.** The journey depends on LLM output
   and is not deterministic: on 2026-08-05 a real defect surfaced in exactly 1 run of 5. One
   green run is one sample, not evidence. The script reports the tally and fails if ANY run
   fails.

   ⚠️ **Never report "green" for a multi-tier suite — name the tiers that ran AND the ones
   that did not.** "Gate green" collapses five tiers into one word and has been wrong in both
   directions on this repo.

**The orchestrator MAY NOT make Quinn optional.** Quinn is automatic between phases per `feedback_quinn_review_mandatory_between_phases_never_optional`. The orchestrator MAY decide HOW to triage Quinn's findings AFTER receiving the verdict, but the review itself is unconditional.

### When the next sprint is launched

Replace this entire "Active Sprint" section with the actual sprint plan reference + per-phase DoD. The phase-boundary discipline above is the persistent baseline; sprint-specific items are layered on top.

<!-- sprint-activate-end -->

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

### Search precedence — graphify vs repomix (added 2026-08-03, reconciles with user-scope Rule 38)

**These are two different jobs. Use the one that matches the question.**

| Question shape | Tool | Why |
|---|---|---|
| **Structural** — "what is X", "what calls/references X", "how do A and B connect", "what are the hubs", "blast radius of changing X" | **`graphify` FIRST** | Answers from the AST graph with exact `file:line`. This is what Rule 38's *"codebase-memory MCP where available (which comes first)"* exception is for — a structured index outranks a text sweep. |
| **Textual** — "every occurrence of this string/literal", "which files mention X", auditing for a leaked path, sweeping comments/docs/config | **`repomix` (Rule 38)** | graphify indexes **symbols, not text**. It cannot answer "does this literal appear anywhere" — a string inside a function body is not a node. Raw `grep`/`rg`/`git grep` remain banned: they silently skip NUL-byte files and exit 0. |
| **Single known file + single known string** | native `Read` / `Grep` | unchanged |

**graphify does NOT supersede Rule 38 — it slots in above it for structural questions and is useless for textual ones.** A negative graphify result is NOT evidence a string is absent.

**Practical notes (measured 2026-08-03):**
- `graphify query` defaults to a **~2000-token budget and silently truncates** — a real query returned *"showing 71 of 400 nodes"*. Raise it with `--budget` or narrow with `--context`, and treat a truncated result as incomplete, not as an answer.
- The graph is **code-only** (`--code-only`), so `.ai-docs/` markdown is NOT indexed. Doc questions go to repomix or `search_docs`.
- Rust caveat: Rust is not in graphify's language-specific member-call resolver set. **Type/import/structure/cross-crate edges are strong; call-graph depth through trait dispatch is weaker.** Don't treat an absent call edge as proof there is no caller.
- **Keeping the graph fresh in a Conductor worktree.** The git post-commit auto-rebuild **deliberately does not fire in linked worktrees** (`hooks.py:258-267` — it exits when git-dir ≠ git-common-dir). That guard is correct: the post-commit rebuild is *delta-based*, so in a worktree with no baseline it would write a `graph.json` containing only the commit's changed files — confidently wrong, worse than absent (upstream #1809; the sibling #1810 saw 5 worktrees inflate a graph from 9,400 nodes/10 MB to 210,000/311 MB). Use instead:
  - `graphify watch .` — filesystem daemon, **no worktree guard**, AST-only, 3s debounce. The intended path here.
  - `graphify extract . --code-only` — full rebuild, ~40s, free.
- ⚠️ **Do NOT set `GRAPHIFY_OUT` to a shared path across Conductor workspaces.** That option exists (upstream #686) but assumes worktrees of the *same* code; these workspaces are divergent branches, so a shared graph would reflect whichever rebuilt last. Per-workspace `graphify-out/` (gitignored) is correct.
