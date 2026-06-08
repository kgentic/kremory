<!-- sprint-activate-begin -->
## Active Sprint

**Sprint**: Foundation Sprint Plan — v0.2.0 BYOE + TD Closure (2026-06-08)
**Plan doc**: foundation-sprint-plan-2026-06-08
**Current phase**: E0 — mechanical file-split (TD-001)
**Activated**: 2026-06-08T20:40:00Z

### Phases

| Phase | Scope | Status |
|---|---|---|
| E0 | Mechanical file-split: facade/mod.rs + extraction/mod.rs + ingest.rs into submodules (TD-001) | pending |
| E | BYOE redesign — ExtractorKind + dual-trait + napi Shape B + TD-006 sweep (ADR-039) | pending |
| G | Episode struct + Migration 009 content_hash backfill (ADR-042, TD-003) | pending |
| H | napi BYOM bridge ThreadsafeFunction crash spike — HUMAN-LED (TD-005) | pending |
| C | Tessa 7 CI gates + module-size lint (TD-002) | pending |
| B | TD ledger close + boy-scout sweep + v0.2.0 tag | pending |

### Definition of Done

- [ ] All 7 Tessa CI gates pass on Rust 1.86 (`cargo test --workspace`, `cargo test --workspace --all-features`, `pnpm test`, `cargo clippy`, `cargo fmt`, `cargo build --release`, module-size lint)
- [ ] TD-001, TD-002, TD-003, TD-005 (or ADR-043 escalation), TD-006 closed in tech-debt-register with commit refs
- [ ] TD-007, TD-008, TD-009, TD-011 confirmed closed (pre-sprint; verify at Phase B ledger pass)
- [ ] TD-004 + TD-010 confirmed OPEN with sprint-slot notes updated to reference this plan
- [ ] Foundation sprint plan status → `completed`
- [ ] v0.2.0 tagged + released per ADR-026 per-release discipline
- [ ] v0.1.1 dream-phase impl first task unblocked (Phase trait scaffolding can begin)

### Out-of-sprint deferrals (autonomous loop MUST reject)

- TD-004 — `episode_tags` junction table (v0.1.7 entity_edges cycle gate)
- TD-010 — Pass 0 shape validator placeholder gap (v0.1.9 Pass 0 gate)
- v0.1.1 dream-phase implementation (depends on this sprint; starts after)
- aidocs vNext napi integration (consumer-side; after Option E surface stable)

### Phase-boundary discipline

- **Recovery tags**: `post-E0-2026-06-08`, `post-E-2026-06-08`, `post-G-2026-06-08`, `post-H-2026-06-08`, `post-C-2026-06-08` created after each phase merge.
- **Pre-commit gate**: `/quality-gate` MUST run before every phase commit per CLAUDE.md non-negotiable rule.
- **Quinn adversarial review**: `/ship-build-review` (Sonnet) MUST run after every phase impl + before tagging per `feedback_quinn_review_before_every_commit`.
- **Phase H is HUMAN-LED**: pause autonomous loop at Phase H slot, defer to user `/ship-spike` with 1.5d cap.

> Managed by `/ship-sprint-activate foundation-sprint-plan-2026-06-08`. Remove this section with `/ship-sprint-complete`.
<!-- sprint-activate-end -->
