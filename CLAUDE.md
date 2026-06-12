<!-- sprint-activate-begin -->
## Active Sprint

**Sprint**: v0.2.3 — Sink Callsite Wiring + Event-Arch
**Status**: ACTIVE — branch `main` @ `1a802d7`, /ship-build launching (diamond topology)
**Spec**: `.ai-docs/specs/v0-2-3-impl-spec-2026-06-12.md` (readiness gate PASS ≥85%)
**Arch spec**: `.ai-docs/specs/v0-2-3-sink-wiring-arch-spec-2026-06-12.md`
**Test strategy**: `.ai-docs/test-strategy/v0-2-3-sink-wiring-test-strategy-2026-06-12.md`
**ADR**: `adr-052-sink-callsite-wiring-and-event-arch-2026-06-12` (accepted; Vera 5 MEDs resolved; Tessa PASS)
**Effort envelope**: 1-2 AI-days, $20-40

**Last shipped — v0.2.2 (ADR-051 GLiNER-to-background unified hot path)**, tagged + pushed:

| Phase | Commit | Scope |
|---|---|---|
| Phase 1 | `05eef75` | Migration 015a — episode_processing_status column |
| Phase 2 | `f3b8f75` | run_verify_stage owns GLiNER + verify_batch + Stage 3 write |
| Phase 3 | `bb105e8` | worker_loop wires run_verify_stage — hot path latency lands |
| Phase 4 | `7fbad81` | Memory::with_await_extraction + Memory::wait_for_processing |
| Phase 5 | `1a802d7` | Observability hardening + Quinn deferred MED fold-in |

All 5 phases shipped with zero `#[allow]` band-aids; every Quinn finding cause-fixed.

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

**Workspace baseline at v0.2.3 start**: 627 passing tests / 1 pre-existing TD-C fail (`c1_module_exists_under_500_loc` — consistency_check.rs at 1234 LoC). Clippy clean.

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

- TD-C — `c1_module_exists_under_500_loc` failing (consistency_check.rs at 1234 LoC). Paired with ADR-050 split sprint. Pre-existing baseline, not a new regression.
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
