<!-- sprint-activate-begin -->
## Active Sprint

**Sprint**: v0.1.2 Empirical-Proof Sprint — TD-035 + TD-036 + ADR-047 Pass 4 (consistency_check)
**Plan doc**: `v0-1-2-empirical-proof-sprint-plan-2026-06-09`
**Current phase**: A — Compile-spike + setup + baseline (in progress)
**Activated**: 2026-06-09

### Phases

| Phase | Scope | Status |
|---|---|---|
| A | Compile-spike (ADR-047 dyn-compat) + Migration 013/012b spec finalisation + sprint setup + baseline gate | in progress (A1 ✓ compile-spike PASS commit 716879c; A3 ✓ baseline-verdict doc filed; A2+A4 PENDING) |
| B | Migration 013 (forward — `entity_type_source = 'DreamPass4'` + `dream_pass4_audit` table) + Migration 012b (downgrade) + idempotency tests | pending |
| C | `crates/kremory/src/core/dream/consistency_check.rs` core module (ConsistencyCheckOpts, run_consistency_check, embed-prefilter gate, Confirm/Reject/Modify schema, cap-overflow guard, 8 observability counters, audit-table writes) | pending |
| D | TD-036 fixture (`mis_typed_high_conf`) + τ calibration sweep + **RISK-001 LOAD-BEARING acceptance gate** (gemma4-e2b:latest ≥5pt precision lift) | pending |
| E | TD-035 fixtures (`catch_all_seed` + `low_confidence_seed`) + benchmark + Pass 2 ↔ Pass 4 co-existence verification | pending |
| F | Engine + DreamOpts integration + Pass 2/4 ordering tests + observability surface verification | pending |
| G | TD ledger close + README rewrite (consumer-profile two-track) + v0.1.2 tag candidate prep + signed-tag push | pending |

### Definition of Done (sprint-level)

- [ ] All 7 phases pass per-phase DoD criteria (47 mechanical checks total) per the plan
- [ ] `/quality-gate` PASSES between every phase commit
- [ ] `cargo test --workspace --all-features` passes (compile + run) on every phase commit
- [ ] No new test failures introduced; baseline preserved (604 passed / 2 failed --all-features as of 2026-06-09 commit c36af5c — 2 failures are TD-012 pre-existing v0.1.9 carry-over, EXPLICITLY out-of-scope per R19)
- [ ] TD-034 + TD-035 + TD-036 status transitions logged in tech-debt-register with commit refs
- [ ] ADR-047 NOT modified during implementation (cycle 1 amendments folded; ratified-only beyond cycle 2 if surfaced)
- [ ] **RISK-001 acceptance gate PASS**: gemma4-e2b:latest ≥5pt absolute lift on TD-036 fixture (LOAD-BEARING)
- [ ] v0.1.2 release tag cut after Phase G with signed annotation including honest framing block

### Out-of-sprint deferrals (autonomous loop MUST reject)

- TD-005 — napi BYOM bridge crash (v0.1.9 binding-layer cleanup)
- TD-012 — background_integration deferred items (v0.1.9 — pre-existing carry-over)
- TD-016 — Ziad fixture text gap (opportunistic / v0.1.9)
- TD-029 — aidocs status-frontmatter bug (file upstream)
- TD-030 — Machine-checkable enum naming constraint (v0.1.2 mechanical; deferred IF time pressure)
- TD-032 — Drift arm Phase E future amendment (open per ADR-046 Option E)
- TD-034 — GLiNER structural-miss handling (deferred to v0.1.3+ unless trivially co-located)
- Pass 2 deprecation evaluation (v0.1.3+ explicit ADR per ADR-047 Pass 2/4 boundary)
- Predicate-shape heuristic Pass 4 sub-step (candidate B from ADR-047 — explicit revisit trigger per ALT-001)
- Frontier-model verify benchmark (PRIMARY local gates ratification; frontier is secondary documentation-only)

### Phase-boundary discipline (non-negotiable per-phase exit gate)

Every phase A through G is COMPLETE ONLY when ALL phase-specific DoD criteria PASS **AND** all three universal exit gates PASS in this order:

1. **`/quality-gate` PASS** — typecheck + lint + test + build per CLAUDE.md "Pre-Commit Gate NON-NEGOTIABLE". No band-aids — cause-fix any failure per Rule 8.
2. **Quinn `/ship-build-review` PASS or CONCERNS-resolved** — Sonnet adversarial review over the phase's aggregate diff. HIGH findings BLOCK; MEDIUM findings folded into the same PR or follow-up; LOW findings filed as new TDs OR rolled into boy-scout sweep per `feedback_boy_scout_includes_quinn_low_findings`.
3. **LLM integration smoke PASS** — phase-specific real-LLM end-to-end verification (per CLAUDE.md Rule 10 verify-before-stating). Per-phase tests `cargo test --workspace --features llm-integration -- --ignored <phase-specific test names>` against live `OLLAMA_HOST` running `gemma4-e2b:latest` per `tests/llm_integration.rs:1-25` SoT. If OLLAMA_HOST unavailable: orchestrator surfaces ONE HITL acknowledgement — never silent skip.

**The orchestrator MAY NOT make Quinn optional.** Quinn is automatic between phases per `feedback_quinn_review_mandatory_between_phases_never_optional`. The orchestrator MAY decide HOW to triage Quinn's findings AFTER receiving the verdict, but the review itself is unconditional.

- **Stop conditions** (7 total): see plan §Stop Conditions — autonomous loop halts on any of them per user's max-autonomy authorisation
- **RISK-001 LOAD-BEARING acceptance criterion**: Phase D D5 MUST measure `gemma4-e2b:latest` precision lift ≥5pt on TD-036 fixture to ratify ADR-047. FAIL → sub-decision (i)/(ii)/(iii)/(iv) revision before Phase E.
- **Module discipline**: `consistency_check.rs` MUST stay <500 LoC at commit per `feedback_split_files_before_adding_when_over_500_loc`
- **Observability per Rule 19**: every phase ships its own counters inline; NOT a sweep phase

### Per-Phase LLM Integration Gate

Per CLAUDE.md Active Sprint discipline + `feedback_verify_model_sot_every_citation_not_just_first_time`:

| Phase | Real-LLM smoke tests required |
|---|---|
| A | None (compile-spike + setup only) |
| B | None (DB migration only) |
| C | `consistency_check_schema_parses` (10× real call, 100% direct parse on gemma4-e2b:latest) |
| D | `benchmark-pass4` on mis_typed_high_conf fixture (RISK-001 LOAD-BEARING) + τ-calibration-sweep |
| E | `benchmark-pass0-pass2-pass4` on catch_all_seed + low_confidence_seed |
| F | `dream_run_full_pass_ordering` real-LLM integration test |
| G | None (ledger + docs + tag) |

### Implementation-readiness verdict (Loop 1)

PASS at ~85% aggregate (2026-06-09). See `.ai-docs/lessons/2026-06-09-v0-1-2-readiness-gate-verdict.md`.

3 dimensions remain inherently-empirical and gated to Phase D D5 RISK-001 acceptance criterion (LLM-verify reliability, τ calibration, Pass 2/4 co-existence). 95% pre-impl is not achievable by category; Phase D D5 IS the empirical-proof mechanism.

### Autonomous chain context (per user 2026-06-09 authorisation)

- **Max-autonomy mode**: push + tag autonomously per user's explicit override of Architectural HITL Gates for THIS sprint scope only
- **No HITL unless absolutely necessary**: re-confirm only on Stop Condition trigger OR irreversible action outside sprint scope
- **Cost cap**: $150 (current spend ~$25-30 post-Loop-1)
- **Skill suite enforcement**: Vera + Quinn + Tessa + /quality-gate all gated; fix all issues before next phase per `feedback_consult_full_skill_suite_in_autonomous_mode`
- **Spawn-tax awareness**: direct execution when output small + predictable; delegate when large + unpredictable

> Sprint scaffolding installed manually (skill `ship:ship-sprint-activate` had detection bugs per `2026-06-09-ship-sprint-activate-dogfood-findings.md`). Remove this section after v0.1.2 ships.
<!-- sprint-activate-end -->
