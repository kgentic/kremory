<!-- sprint-activate-begin -->
## Active Sprint

**Sprint**: v0.2.4 — Dream-Pass Crash-Safety (ADR-050) — **build COMPLETE, UNPUSHED**
**Status**: v0.2.4 implemented on branch `kgentic-dev/restore-session` (8 commits unpushed). Workspace GREEN (1164 tests / 0 fail; clippy `-D warnings` clean; fmt clean). **Blocked on push** — `kgentic` org GitHub Actions billing dead → no CI validation; awaiting billing restore before tag/push `v0.2.4`. **NEXT: tech-debt clearance sprint** (TD-042 Tier 2 + audit backlog) — see `.ai-docs/tech-debt/tech-debt-register.md`.
**Spec**: `.ai-docs/specs/v0-2-4-impl-spec-2026-06-12.md` (readiness gate PASS; both compile-spikes discharged)
**Arch spec**: `.ai-docs/specs/v0-2-4-crash-safety-arch-spec-2026-06-12.md`
**ADR**: `adr-050-dream-pass-crash-safety-and-idempotency-2026-06-11` (SHIPPED — was DEFERRED, now done)

**Last shipped — v0.2.4 (ADR-050 dream-pass crash-safety)**, committed, UNPUSHED:

| Phase | Commit | Scope |
|---|---|---|
| Groundwork | `ec766e4` | compile-spikes (`idempotency.rs` + `token_counting.rs`) + cross-crate fixes |
| Phase 1 | `d128b09` | Migration 016 crash-safety schema + 015b downgrade |
| Phase 3 | `85e4fce` | idempotency-key + checkpoint resume + cooldown + is_dream_generated |
| Phase 4 | `8ec711f` | budget tracking (TokenCountingChatProvider wiring) |
| Phase 5 | `1ec121e` | SkippedIdempotent + on_worker_resumed sink events |

Quinn PASS every phase (90/93/94/96). Zero `#[allow]` band-aids added to `src/`. Plus tech-debt commits: `6fa7229`/`571c623` (rust lint conventions), `e6a0fda` (TD-042 Tier 1 — 35 redundant test allows removed).

> ⚠️ **The v0.2.3 sprint detail BELOW is SUPERSEDED** (predates the v0.2.4 build). Treat as historical. The earlier "1 pre-existing fail" baseline claim is STALE — workspace is now fully green. A full Active-Sprint refresh is part of the tech-debt clearance sprint. The "Phase-boundary discipline" subsection further down remains the persistent baseline and still applies.

### v0.2.3 phase plan (per impl spec §6)

| Phase | Topology | Files | Effort |
|---|---|---|---|
| 1 — IngestStatus::EntitiesReady + from_sql_status bridge | sequential | core/sink.rs, tests/sink_trait_shape.rs | 30 min |
| 2 — BackgroundIngestor.sink field + Memory::with_sink() builder | sequential | core/background/, facade/, memory/events.rs | 45 min |
| 3 — Background pipeline callsite wiring (11 fire-sites) | **diamond — 3 parallel sub-agents (file-disjoint)** | verify_stage.rs / deferred_pipeline.rs / pipeline.rs | 2-3h |
| 4 — BatchPhase2Complete + BatchProgress tracker | sequential | NEW batch_tracker.rs, background/, facade/ | 45 min |
| 5 — Observability surfaces (deferred_queue_depth gauge, callback_duration_ms histogram) | sequential | deferred_pipeline.rs, worker_loop.rs, batch_tracker.rs | 30 min |
| 6 — Test surface | sequential | NEW tests/helpers/recording_sink.rs, NEW tests/sink_wiring.rs, NEW tests/sink_wiring_integration.rs | 2h |
| 7 — CI gates + doc-comment polish | sequential | scripts/check-dual-emit.sh, NEW scripts/check-sink-callsite-coverage.sh, memory/events.rs | 1h |

**Workspace baseline (corrected 2026-06-16, P1 hardening)**: TD-C now CLOSED (consistency_check.rs split → c1 PASS). The "1 pre-existing fail" baseline was wrong — stash-tests on clean HEAD `114b158` confirm **two** pre-existing fails remain: (1) `with_facts_empty_vec_equivalent_to_no_facts` (TD-013, kremory) and (2) `napi_surface_matches_substrate_or_skip_list` (kremory-napi — `parity-skip.toml` 103 entries > its own 100 cap). The napi one is invisible to `cargo test -p kremory` (cross-crate gate). Clippy clean; `cargo fmt --check` NOT hard-enforced (main has drift in `sink_fires_through_ingest.rs`).

### v0.2.3 DoD highlights (per impl spec §2)

1. All 14 sink fire-sites wired (arch spec §3.1 catalogue)
2. `IngestStatus::EntitiesReady` variant added (`#[non_exhaustive]` preserved)
3. `from_sql_status()` bridge at `core/sink.rs` (one-way SQL → enum)
4. `BackgroundIngestor.sink` field + `Memory::with_sink()` builder
5. Triple-emit pattern at every callsite (sink + `metrics::counter!/histogram!` + `tracing::*` within 5 source lines — Alt 4 unified enum REJECTED per ADR-2026-05-20 D1)
6. D7 cardinality discipline (entity_id / episode_id / batch_id NEVER as metric labels)
7. New monitoring: `rql.background.deferred_queue_depth` gauge (Vera MED-04 OOM early-warning)
8. New observability: `kremory.sink.callback_duration_ms` histogram (G7 slow-consumer detection)
9. CI gates: `check-dual-emit.sh` (8 new entries) + NEW `check-sink-callsite-coverage.sh`
10. Test surface: `RecordingSink` helper + `sink_wiring.rs` (L1/L2 ≥15 tests) + `sink_wiring_integration.rs` (L3 ≥6 tests)
11. `events.rs` doc-comment polish: MED-02 inline-path zero-events limitation, MED-05 ADR-050 idempotency forward-compat, MED-01 normative Deduplicating fire condition, D4 thread-context contract

### Out of scope (explicitly excluded per impl spec §1)

- ADR-050 crash-safety + idempotency cluster (v0.2.4+)
- napi-rs binding parity (v0.3.0)
- `on_community_updated` wiring (depends on stable dream-pass)
- Substrate-owned bounded backpressure queue (deferred per D3)
- `cargo-nextest` migration
- Inline-path (`run_in_background=false`) sink wiring — known asymmetry (Vera MED-02)
- `engine_handle.rs::graph_ingest_episode(run_in_background=true)` tokio-spawn path sink wiring

### Sprint-specific HITL boundaries (per impl spec §9)

| Boundary | Trigger | Action |
|---|---|---|
| Mid-Phase 2 plumbing | trait-object / lifetime design beyond spec | surface, do not guess |
| Mid-Phase 3 | Quinn HIGH on aggregate diff | halt, fix, re-run Quinn |
| Mid-Phase 4 | BatchProgress concurrency design beyond `Mutex<HashMap>` vs `DashMap` | surface with data |
| Pre-push | tag `v0.2.3` | user confirms tag before push (no autonomous force-push) |
| napi-rs thread-context discovery | Phase 6 L3 NonBlocking deadlock | HALT + escalate (architectural signal on D4 contract) |

### Open architectural decisions

- ✅ **ADR-052 (accepted 2026-06-12)** — Sink callsite wiring + event-arch. This sprint.
- ✅ **ADR-051 (accepted 2026-06-11, shipped 2026-06-12)** — GLiNER-to-background unified hot path.
- ✅ **ADR-048 (accepted 2026-06-11) — scope v0.3.0** — Three-signal local-first consistency check.
- ✅ **ADR-049 §5.5 SLA** — superseded by ADR-051 §5.5 SLA Cascade Amendment.
- ⏸ **ADR-050 (drafted, DEFERRED v0.2.4+)** — Dream-pass crash-safety + idempotency cluster. Cross-ADR coupling: crash-resume will retroactively force consumer-side idempotency for sink callbacks (documented in ADR-052 Consequences + Phase 7 doc-comment).

### Strategic posture docs (load-bearing for future planning)

- `cloud-readiness-posture-2026-06-11` — passive cloud-readiness, no speculative engineering. Apply 30-second self-check at every future ADR/spec.
- `queue-and-background-worker-research-2026-06-11` — Recommended hybrid P1+P2 (tokio mpsc + apalis SQLite) when durability becomes a requirement.

### Closed (carried over)

- ✅ TD-A — 4× never_list_* test fixture drift
- ✅ TD-B — 2× background_integration LLM mock drift
- ✅ ADR-051 ratification + §5.5 cascade
- ✅ ADR-048 ratification (v0.3.0 scope)
- ✅ v0.2.2 sprint shipped + tagged + pushed (all 5 phases, zero `#[allow]` band-aids)

### Still open

- ✅ TD-C — CLOSED 2026-06-16 (P1a, commit `01231d6`): consistency_check.rs split into mod/verify/audit (each <500 LoC); `c1_module_exists_under_500_loc` PASS. Done ahead of the ADR-050 sprint.
- TD-013 — `with_facts_empty_vec_equivalent_to_no_facts` failing (empty-vec skip increments skip_extraction counter when it should not). Pre-existing baseline, confirmed via stash-test; scheduled for P2 (fix the bug per Rule 8, not skip).
- napi parity-cap — `napi_surface_matches_substrate_or_skip_list` fails: `parity-skip.toml` has 103 entries > its own sanity cap of 100. Pre-existing; binding-layer drift (per substrate-first posture, binding parity work defers while substrate evolves). Fix = prune skip-list by refining `tracked_impl_types()`/`tracked_struct_types()` scope, not inflating the cap.
- TD-005 / TD-012 / TD-016 / TD-029 / TD-030 / TD-032 / TD-034 — pre-existing, out-of-scope per prior deferrals.
- TD-H (informal) — RISK-001 methodology hardening (multi-run mean ± SD).
- Vera MED-04 deferred-queue OOM monitoring — spec'd, lands in v0.2.3 Phase 5.
- Recurring git-add-untracked-test pattern (3× in v0.2.2) — worker-brief discipline TD for ship-build skill update.
- v0.2.2 gah backup branch `backup-after-hours-2026-06-12T06-29-57` — safe to delete after CI green.

### Phase-boundary discipline (preserved across sprints)

Every phase commit MUST satisfy these gates in order — they apply to any sprint type unless explicitly waived in the sprint plan:

1. **`/quality-gate` PASS** — typecheck + lint + test + build per CLAUDE.md "Pre-Commit Gate NON-NEGOTIABLE". No band-aids — cause-fix any failure per Rule 8.
2. **Quinn `/ship-build-review` PASS or CONCERNS-resolved** — Sonnet adversarial review over the phase's aggregate diff. HIGH findings BLOCK; MEDIUM findings folded into the same PR or follow-up; LOW findings filed as new TDs OR rolled into boy-scout sweep per `feedback_boy_scout_includes_quinn_low_findings`.
3. **LLM integration smoke PASS** — phase-specific real-LLM end-to-end verification when relevant (skip for pure-docs sprints).

**The orchestrator MAY NOT make Quinn optional.** Quinn is automatic between phases per `feedback_quinn_review_mandatory_between_phases_never_optional`. The orchestrator MAY decide HOW to triage Quinn's findings AFTER receiving the verdict, but the review itself is unconditional.

### When the next sprint is launched

Replace this entire "Active Sprint" section with the actual sprint plan reference + per-phase DoD. The phase-boundary discipline above is the persistent baseline; sprint-specific items are layered on top.

<!-- sprint-activate-end -->
