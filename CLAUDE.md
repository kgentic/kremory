<!-- sprint-activate-begin -->
## Active Sprint

**Sprint**: v0.1.1 Dream-Impl Sprint — Multi-tier Architecture + Pass 0 + Reclassify
**Plan doc**: `v0-1-1-dream-impl-sprint-plan-2026-06-09`
**Current phase**: A — Migration 010 + source-tier contract
**Activated**: 2026-06-09

### Phases

| Phase | Scope | Status |
|---|---|---|
| A | Migration 012 (source-tier columns + legacy backfill — drift view deferred to Phase E per ADR-046 Amendment); `RawEntityIntegerId.confidence`; per-extractor source-tier contract; symmetric `ConsumerPinned` write (subject + object_id) | **complete** (a57fb6f, Quinn PASS 89/100) |
| B | TD-028 Phase 1 pull-shape registry-read derivation | pending |
| C | Dream API surface: `Engine::run_dream_pass_sync`, `DreamOpts`, `ghost_episodes`, `assert_entity_type`, `Mutex<()>` serialization | pending |
| D | Dream Pass 0 implementation (ADR-037): clustering + LLM proposal call + anti-redundancy gate | pending |
| E | Dream Pass 2 reclassify (ADR-046 Option E — 2-arm SELECT catch_all_cascade + low_confidence, drift arm deferred per Amendment 2026-06-09): confidence-aware source-tier write | pending |
| F | Observability + ergonomics: rate limits + concurrency docs + within-episode contradiction pre-check | pending |
| G | TD-017 empirical hatch benchmark (gemma4-e2b + cloud-LLM) | pending |
| H | TD ledger close + boy-scout sweep | pending |

### Definition of Done (sprint-level)

- [ ] All 8 phases pass per-phase DoD criteria (48 mechanical checks total) per the plan
- [ ] `/quality-gate` PASSES between every phase commit
- [ ] `cargo test --workspace --all-features` passes (compile + run) on every phase commit
- [ ] No new test failures introduced; baseline preserved (80 passed default-features as of 2026-06-09)
- [ ] TD-017..TD-029 status transitions logged in tech-debt-register with commit refs
- [ ] ADR-044, ADR-045, ADR-046, ADR-037 NOT modified during implementation (ratified-only)
- [ ] v0.1.1 release tag cut after Phase G empirical hatch verdict captured

### Out-of-sprint deferrals (autonomous loop MUST reject)

- TD-005 — napi BYOM bridge crash (v0.1.9 binding-layer cleanup)
- TD-016 — Ziad fixture text gap (opportunistic / v0.1.9)
- TD-029 — aidocs status-frontmatter bug (file upstream)
- v0.2.x `ExtractorKind::GlinerLlm` deletion (gated by Phase G empirical hatch)
- aidocs vNext napi integration (consumer-side; after v0.1.1 stable)

### Phase-boundary discipline (non-negotiable per-phase exit gate)

Every phase A through H is COMPLETE ONLY when ALL phase-specific DoD criteria PASS **AND** both universal exit gates PASS in this order:

1. **`/quality-gate` PASS** — typecheck + lint + test + build per CLAUDE.md "Pre-Commit Gate NON-NEGOTIABLE". No band-aids — cause-fix any failure per Rule 8.
2. **Quinn `/ship-build-review` PASS or CONCERNS-resolved** — Sonnet adversarial review over the phase's aggregate diff. HIGH findings BLOCK; MEDIUM findings folded into the same PR or follow-up; LOW findings filed as new TDs.

**The orchestrator MAY NOT make Quinn optional via HITL.** Quinn is automatic between phases. The orchestrator MAY ask the user how to triage Quinn's findings AFTER receiving the verdict, but the review itself is unconditional. Per `feedback_quinn_review_before_every_commit` (2026-05-31 TD-012 diamond fabricated "595 tests PASS" while build was broken) — tests-PASS alone is insufficient at phase boundaries.

- **Stop conditions** (7 total): see plan §Stop Conditions — autonomous loop halts + HITLs on any of them
- **Phase G empirical hatch**: per-provider INDEPENDENT pass; aggregate-pass FORBIDDEN per ADR-044 §5
- **Source-tier contract**: per Migration 012 detail spec §1.2 — all entity insert sites MUST pick a value from the authoritative table

### Implementation-readiness verdict

PASS (95.5% aggregate, 2026-06-09). See `.ai-docs/lessons/2026-06-09-v0-1-1-readiness-gate-verdict.md`.

> Sprint scaffolding installed manually per `2026-06-09-ship-sprint-activate-dogfood-findings.md` (skill had 4 bugs). Remove this section after v0.1.1 ships.
<!-- sprint-activate-end -->
