---
title: "Public docs + API-surface audit — residual risk & contingency register"
type: risk-register
status: draft
created: 2026-09-07
spec: docs/specs/public-docs-and-api-surface-audit/spec.md
plan: docs/specs/public-docs-and-api-surface-audit/plan.md
---

# Residual risk & contingency register

Required at this spec's tier (high-stakes — phased work, per the tier
elevation recorded in `spec.md`'s Tier + gate rationale section). Each row is a
gap that cannot be fully answered upfront; each gets detection, containment,
recovery, and a fallback, so nothing is merely listed.

| # | Risk | Probability | Impact | Detection | Containment | Recovery | Fallback / stop condition |
|---|---|---|---|---|---|---|---|
| R1 | Phase 0 compile harness's scaffold contract (RULE-001) is wrong in a way not caught by the design review — e.g. the per-section grouping still leaves some blocks unresolvable | Medium | High — reproduces the exact "audit tool itself is broken" failure this whole spec exists to prevent | First harness run reports a failure rate wildly inconsistent with a spot-check of 5 random blocks read by hand | Do not trust the harness's pass/fail numbers until a human (or a second, independent pass) spot-checks ≥10% of reported failures against the actual doc prose | Adjust the prelude/grouping rule, re-run, re-spot-check | STOP condition: if 2 consecutive harness runs need scaffold-contract changes, halt Phase 0 and treat the mechanism itself as unproven — do not proceed to Phase 1 on an unstable harness |
| R2 | A Phase 2 fix requires a breaking API change bigger than the "additive accessor" pattern TD-231 set as precedent — e.g. removing/renaming a public item | Low | High — a breaking rename affects both the Rust crate's semver contract and the napi 1:1 parity gate simultaneously | The fix diff itself: if it touches a `pub` item's name or removes a `pub` item rather than adding one | Implement and commit locally per RULE-006 (never publish); do not let it merge into a "just docs" mental model | Flag explicitly in the SC-003/RULE-006 findings table with severity HIGH for maintainer go/no-go — do not silently absorb it as a routine Phase 2 fix | STOP condition: any finding requiring a breaking rename/removal pauses that specific finding for explicit maintainer sign-off before implementation continues; other findings proceed |
| R3 | RULE-013's napi-parity-skip-cap raise becomes routine (every Phase 2 fix "needs" a cap raise) rather than exceptional, silently eroding the parity gate's meaning | Medium | Medium — the parity gate exists specifically to keep drift visible; a gate that's raised on every PR stops being a gate | Count of RULE-013 invocations across Phase 2 | Cap raises require the one-line rationale convention (already established, e.g. `parity-skip.toml` history) — never a silent raise | If ≥3 raises occur in this spec's Phase 2, stop and ask whether the underlying pattern (e.g. every Tier-2-builder knob needing its own skip entry, ADR-030 Form B) should be fixed structurally instead of one skip-list entry at a time | STOP condition: 3rd raise in this spec triggers an explicit note in the conformance report recommending a structural fix, not a 4th silent raise |
| R4 | Docusaurus build (Phase 5) works locally but the maintainer's later deploy (Phase 6, separately gated) surfaces problems this spec's local-build exit criterion (RULE-009/SC-005) can't see — e.g. base-path/CDN-relative-link issues that only manifest once actually hosted | Medium | Low — deploy is explicitly out of scope and separately gated; this spec cannot fully de-risk it | N/A until Phase 6 runs (outside this spec's scope) | Use relative links and Docusaurus's own recommended path conventions during Phase 5, per its docs, rather than hardcoded absolute URLs — reduces but does not eliminate deploy-time risk | Phase 6 (separate, maintainer-gated) is where this actually gets fixed if it surfaces | Accepted residual risk — explicitly not this spec's problem to fully close, per Non-goals; noted here so it isn't forgotten between now and Phase 6 |
| R5 | The getting-started transcript (Phase 3, RULE-008) is run once, passes, and then drifts as Phase 2 lands more fixes afterward — an ordering hazard given Phase 3 can run in parallel with Phase 4 per the plan's sequencing note | Medium | Medium — a stale getting-started guide is exactly the kind of doc/code mismatch this audit is meant to eliminate | Re-run the transcript command sequence (cheap — it's `cargo add kremory` + a few calls) as the very last step before Phase 5 content migration, not just once mid-Phase-3 | Treat the transcript as provisional until re-run post-Phase-2 | Re-capture the transcript if any Phase-2 commit lands after the first capture and touches a surface the transcript exercises | STOP condition: do not migrate the getting-started doc into the Docusaurus site (Phase 5) on a transcript captured before the LAST Phase-2 commit |
| R6 | Autonomous "yolo mode" execution drifts into one of the project's standing HITL gates (`cargo publish`, `git push`, release tag, paid benchmark run) without a deliberate stop, because a later step in a long autonomous chain loses track of the constraint stated at the start | Low | High — these are explicit standing project gates, not this spec's to waive | Every commit/action taken during implementation is checked against the 4-item gate list before execution, not just at spec-authoring time | Local commits only; no `git push`, no `cargo publish`, no tag, no paid benchmark, for the entire duration of this spec's implementation | If any such action is about to be taken, stop and surface it explicitly to the maintainer rather than proceeding | Hard stop — no fallback needed, these are binary gates with no "close enough" |

## Notes

- R3 exists specifically because Finding 5 of the spec's adversarial review
  surfaced a real, already-demonstrated instance of the napi-parity cap being
  hit by exactly this class of fix (TD-231 itself needed a skip-list entry).
  This register formalizes that as an ongoing risk to watch across the whole
  spec, not a one-off.
- This register needs only `spec.md` and `plan.md`'s phase list, per specflow's
  own `risk-contingency-register` skill description — no `tasks.md` dependency.
