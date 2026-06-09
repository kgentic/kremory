<!-- sprint-activate-begin -->
## Active Sprint

**Sprint**: v0.1.1 Dream-Impl Sprint — Multi-tier Architecture + Pass 0 + Reclassify
**Plan doc**: `v0-1-1-dream-impl-sprint-plan-2026-06-09`
**Current phase**: A — Migration 010 + source-tier contract
**Activated**: 2026-06-09

### Phases

| Phase | Scope | Status |
|---|---|---|
| A | Migration 010 (source-tier columns + drift_detection_view + legacy backfill); `RawEntityIntegerId.confidence`; per-extractor source-tier contract; `ConsumerPinned` write | pending |
| B | TD-028 Phase 1 pull-shape registry-read derivation | pending |
| C | Dream API surface: `Engine::run_dream_pass_sync`, `DreamOpts`, `ghost_episodes`, `assert_entity_type`, `Mutex<()>` serialization | pending |
| D | Dream Pass 0 implementation (ADR-037): clustering + LLM proposal call + anti-redundancy gate | pending |
| E | Dream Pass 2 reclassify (ADR-046 unified scope): 3-arm SELECT + confidence-aware source-tier write | pending |
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

### Phase-boundary discipline

- **Pre-commit gate**: `/quality-gate` MUST run before every phase commit per CLAUDE.md non-negotiable rule
- **Quinn adversarial review**: `/ship-build-review` MUST run after every phase impl per `feedback_quinn_review_before_every_commit`
- **Stop conditions** (7 total): see plan §Stop Conditions — autonomous loop halts + HITLs on any of them
- **Phase G empirical hatch**: per-provider INDEPENDENT pass; aggregate-pass FORBIDDEN per ADR-044 §5
- **Source-tier contract**: per Migration 010 detail spec §1.2 — all entity insert sites MUST pick a value from the authoritative table

### Implementation-readiness verdict

PASS (95.5% aggregate, 2026-06-09). See `.ai-docs/lessons/2026-06-09-v0-1-1-readiness-gate-verdict.md`.

> Sprint scaffolding installed manually per `2026-06-09-ship-sprint-activate-dogfood-findings.md` (skill had 4 bugs). Remove this section after v0.1.1 ships.
<!-- sprint-activate-end -->
