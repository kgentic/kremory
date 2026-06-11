<!-- sprint-activate-begin -->
## Active Sprint

**Status**: BETWEEN SPRINTS — handoff state on branch `spike/c6-async-gate-feasibility-2026-06-10`.

**Last work** (chronological, this branch):

| Phase | Commit | Scope |
|---|---|---|
| v0.2.0 Phase A foundational refactor | `6a023bf` + `76b5710` | ADR-049 ratified + Engine::ingest split + verify_batch promotion (pre-this-session) |
| v0.2.0 Phase B-prep Phase A | `ae6de1b` | Tier 0 doc fixes + benchmark correction + fmt sweep |
| v0.2.0 Phase B-prep Phase B | `2c496bf` | Tier 1 mechanical wins (7/10, 3 deferred) + Quinn fold-in |
| v0.2.0 Phase B-prep Phase C | `598c0da` | T2.1 background.rs split + T2.2 confidence gate + Quinn fold-in |
| v0.2.0 Phase B-prep Phase D | `db305be` | T=0 sweep + retro + benchmark addendum |
| O11y Tier 0 sprint | `6222156` | phase1_ner_bench + sweep metrics + llm_tokens_bench |
| Strategy docs | `aa96ff1` | queue-and-background-worker research + cloud-readiness posture |
| Docs sprint (current) | TBD | ADR-049 amendment + ADR-051 draft + this CLAUDE.md update + tech-debt-register |

**Workspace state**: 627 passing tests / 1 pre-existing TD-C fail (`c1_module_exists_under_500_loc` — consistency_check.rs at 1234 LoC, paired with ADR-050 split sprint). Clippy clean. Branch ready to PR or to enter next sprint.

### Next-sprint candidates (in priority order, not yet locked)

| Candidate | Effort | Why now |
|---|---|---|
| **PR opening + v0.2.0 milestone tag prep** | ~30 min | Six commits ready; merge or rebase decision needed |
| **ADR-049 §5.5 SLA amendment + ADR-051 ratification** | ~15 min lightweight OR ~30 min `/ship-decision` | Empirical refutation of <100ms target needs formal close. ADR-051 captures GLiNER-to-background unified hot-path; pending lightweight ratification or `/ship-decision` cycle. |
| **ADR-050 sprint — dream-pass crash-safety + idempotency** | ~1-2 days | Bundles TD-A/B/C remediation gaps: T1.6 cooldown-on-success, T1.10 budget tracking, audit findings #90/#92/#96/#98 (op_checkpoints, content-hash idempotency, is_dream_generated, cooldown-only-on-success) + consistency_check.rs split |
| **`/ship-decision` on ADR-048 ratify-or-defer** | ~15 min | Now data-backed with 2 viable Path-β candidates (qwen2.5:14b + qwen3.6:35b-mlx per T=0 sweep) |
| **GLiNER profiling spike** (orthogonal to ADR-051) | ~half day | ONNX runtime config, entity_types pruning, smaller GLiNER quant — could yield 200-300ms range independent of architectural move |
| **Tier 1 D1.1/D1.2 — multi-size scaling sweep** | ~10h | Now that phase1_ner_bench + llm_tokens_bench exist; gives scaling matrix instead of single-fixture data |
| **Tier 1 D1.3 — real-world GT-annotated fixtures** | ~10h manual | 10-20 real episodes (email/transcript/doc/chat) to validate the synthetic-extrapolation claim |

### Open architectural decisions (formal docs exist; not ratified)

- **ADR-050 (candidate)** — Dream-pass crash-safety + idempotency cluster. Named in `prior-art-adoption-audit-2026-06-10.md` Top 5 finding #3. No ADR doc written yet; would be sprint-spec'd from the audit findings.
- **ADR-051 (proposed)** — GLiNER-to-background unified hot path. Doc at `.ai-docs/adrs/adr-051-gliner-to-background-unified-hot-path-2026-06-11.md`. Pending lightweight ratification OR `/ship-decision` cycle.
- **ADR-048 (proposed)** — Three-signal local-first consistency check. Already exists; ratification pending; now data-backed.

### Strategic posture docs (load-bearing for future planning)

- `cloud-readiness-posture-2026-06-11` — passive cloud-readiness, no speculative engineering. Apply 30-second self-check at every future ADR/spec.
- `queue-and-background-worker-research-2026-06-11` — Rust crate landscape + OSS competitor patterns. Recommended hybrid P1+P2 (tokio mpsc + apalis SQLite) when durability becomes a requirement.

### Closed TDs (this session)

- ✅ TD-A — 4× never_list_* test fixture drift (Phase B-prep Phase A, commit `ae6de1b`)
- ✅ TD-B — 2× background_integration LLM mock drift (Phase B-prep Phase B, commit `2c496bf`)

### Still open

- TD-C — `c1_module_exists_under_500_loc` test failing (consistency_check.rs at 1234 LoC). Paired with ADR-050 split sprint. Pre-existing baseline, not a new regression.
- TD-005 / TD-012 / TD-016 / TD-029 / TD-030 / TD-032 / TD-034 — pre-existing, all out-of-scope per prior sprint deferrals
- TD-H (informal) — RISK-001 methodology hardening (multi-run mean ± SD, currently only T=0 + seed pinned). From O11y sprint retro.

### Phase-boundary discipline (preserved across sprints)

Every phase commit MUST satisfy these gates in order — they apply to any sprint type unless explicitly waived in the sprint plan:

1. **`/quality-gate` PASS** — typecheck + lint + test + build per CLAUDE.md "Pre-Commit Gate NON-NEGOTIABLE". No band-aids — cause-fix any failure per Rule 8.
2. **Quinn `/ship-build-review` PASS or CONCERNS-resolved** — Sonnet adversarial review over the phase's aggregate diff. HIGH findings BLOCK; MEDIUM findings folded into the same PR or follow-up; LOW findings filed as new TDs OR rolled into boy-scout sweep per `feedback_boy_scout_includes_quinn_low_findings`.
3. **LLM integration smoke PASS** — phase-specific real-LLM end-to-end verification when relevant (skip for pure-docs sprints).

**The orchestrator MAY NOT make Quinn optional.** Quinn is automatic between phases per `feedback_quinn_review_mandatory_between_phases_never_optional`. The orchestrator MAY decide HOW to triage Quinn's findings AFTER receiving the verdict, but the review itself is unconditional.

### When the next sprint is launched

Replace this entire "Active Sprint" section with the actual sprint plan reference + per-phase DoD. The phase-boundary discipline above is the persistent baseline; sprint-specific items are layered on top.

> Branch + workspace are in clean handoff state. PR not yet opened. Whoever picks up next has full freedom to choose from the candidates above.
<!-- sprint-activate-end -->
