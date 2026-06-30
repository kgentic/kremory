<!-- sprint-activate-begin -->
## Active Sprint

**No active sprint.** Last shipped: **kremory 0.3.2** (live on crates.io — episodic-edge presence-uniqueness fix). `origin/main` = `fad4d5d` (0.3.2 + TD-053 napi-parity hygiene). The post-v0.2.4 Tech-Debt Clearance sprint and the 0.3.x release line are **DONE and merged to main** — the prior "30 commits ahead / UNPUSHED / v0.2.4 bundle" framing is retired (it predated the 0.3.x releases).

**Current baseline (verified 2026-06-25):**

- Releases **0.3.0 → 0.3.1 → 0.3.2** shipped, tagged, on crates.io; `main` fast-forwarded through all. 0.3.0 yanked (broken default); 0.3.1 + 0.3.2 live.
- **TD-053 (napi parity hygiene) CLOSED + on main** (`a05be15` doc-rot pass, `fad4d5d` boy-scout pass): `parity-skip.toml` = **72 entries ≤ 80 cap**, `cargo test -p kremory-napi --test api_parity` **7/0 GREEN**. The napi 1:1 surface is verified 1:1 (ADR-034 landed); parity-skip now lists only genuinely-absent symbols.
- **One known pre-existing test fail**: `with_facts_empty_vec_equivalent_to_no_facts` (TD-013, kremory) — empty-vec skip increments the skip_extraction counter when it should not. Fix the bug per Rule 8, do not skip. (This is the ONLY standing fail; the old "napi parity-cap fail" was never real — the "103>100" number was stale/fabricated per `feedback_subagent_fabricates_gate_results`.)
- `cargo fmt --check` is NOT hard-enforced (main has drift in `sink_fires_through_ingest.rs`).
- The napi parity test is invisible to `cargo test -p kremory` (cross-crate gate) — run workspace-wide to see it.

**Standing constraints (load-bearing for ANY release / push):**

- **No CI** — `kgentic` org GitHub Actions billing suspended (no restore expected). All 4 workflows (`ci` / `napi-ci` / `release-please` / `eval-canary`) are FULLY COMMENTED OUT (RESTORE header + `git revert` on each). Cross-platform binary builds + crate/npm publishes are **manual** off the M4 Max.
- **Push + release-tag = irreversible HITL gate** — no autonomous push or tag; user confirms each. Clean fast-forward to `main` is the normal land path for hygiene commits.
- **gah convention** — `.gahrc` present (work hours 9–17). Run gah before push to shift unpushed commits out-of-hours; it no-ops when commits are already OOH, and its scope is the unpushed range only (`origin/main..HEAD`). gah refuses on a dirty tree — untracked files from a parallel agent count as dirty (skip gah when already-OOH in that case).

**Open backlog** — `.ai-docs/tech-debt/tech-debt-register.md` (living) is the SoT. Still open at last review: TD-013 (above); TD-043 / TD-045 god-file splits (`graph.rs`, `ingest/pipeline.rs` — PARTIAL, verify current LoC against the register before resuming); TD-005 / TD-012 / TD-016 / TD-029 / TD-030 / TD-032 / TD-034 (pre-existing deferrals); TD-H (RISK-001 multi-run mean±SD hardening); **TD-086 / TD-087 / TD-088 / TD-089 (dream-phase debt cluster, 2026-06-30). HEADLINE = TD-089: live `mem.dream()` runs only 2 of 5 designed passes (discover + reclassify); spec-canonical orchestrator `run_dream_phase_passes` is dead code; aliases + canonicalize orphaned; consistency-check (TD-088) dormant; `PASSES` static drift (TD-086); reclassify maturity-gate decision (TD-087, coupled to TD-088). Crossed threshold to "needs reconciliation spec v2 → /ship-arch-review (Vera) → readiness-gated build". Aliases is ORPHANED not stubbed — earlier "stub" note was wrong.** Verify any TD's status against the register + actual code before acting — register narrative can lag.

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

**The orchestrator MAY NOT make Quinn optional.** Quinn is automatic between phases per `feedback_quinn_review_mandatory_between_phases_never_optional`. The orchestrator MAY decide HOW to triage Quinn's findings AFTER receiving the verdict, but the review itself is unconditional.

### When the next sprint is launched

Replace this entire "Active Sprint" section with the actual sprint plan reference + per-phase DoD. The phase-boundary discipline above is the persistent baseline; sprint-specific items are layered on top.

<!-- sprint-activate-end -->
