---
title: "Dream phase — KEEP or CUT: pre-registered decision rule"
type: decision
status: pre-registered
created: 2026-08-19
tags: [kremory, dream, benchmark, pre-registration, locomo]
---

# Dream phase — KEEP or CUT

**This file is written and committed BEFORE any measurement is taken.** Its whole
purpose is to remove my own freedom to reinterpret the result afterwards. If the
numbers come back ambiguous, the rule below decides — not a fresh argument.

## Why this exists

The dream phase has never been shown to be worth anything end to end. Weeks of
sessions have set out to answer that and instead found defects *on the way to* the
answer — a blind scorer, a vacuous invariant, a guard that never fired, a register
that lied. Each finding was real. The question was never answered.

The only hard data on any part of dream is the TD-222 A/B on the acronym/nickname
pass: **zero benefit on either scorer, ~84 minutes of runtime, and it dissolved
both conversation speakers into the family pets.** That pass is now default-off.

The project owner has authorised ONE bounded session to settle the rest. If it
cannot be answered here, the outcome is CUT by default.

## Design

Paired, single conversation (`conv0`, **n = 199** questions).

Identical across both arms: corpus, binary, models, embedder, recall limit,
`group_id`, scorer code. **The only difference is whether the harness calls
`client.consolidate("creative", namespace)`** (`bench/locomo/harness.py:694`).

- **arm `dream-on`** — consolidation called (current behaviour)
- **arm `dream-off`** — consolidation skipped

Both arms run through `bench/locomo/run_local_ingest.sh`, which is the ONLY free
path: it `env -u`s both Groq chat variables and then asserts on the server's own
boot log that the paid cloud branch did not fire. Cost: £0.

## Primary metric

**qa-gen** (`bench/locomo/qa_eval.py`) — the externally quotable scorer, matched to
the standard LoCoMo answer-generation protocol. The substring scorer is recorded
alongside for continuity with prior runs but is **NOT decisive**; it runs ~17pt
higher on the same run and is order-invariant.

## VALIDITY GATE — checked BEFORE any score is looked at

A null result is only evidence if the lever actually moved something. Verify, in
this order, and stop if either fails:

1. `dream-on` DB has **≥ 1** row in `graph_mutation_log`.
2. `dream-off` DB has **exactly 0** rows in `graph_mutation_log`.

Count via `SELECT recorded_at`-style line counting, never `COUNT(*)` — on the
vector-indexed tables `COUNT(*)` silently returns 0 on populated tables
(SYSTEM-PRIMER gotcha #1). `graph_mutation_log` is not vector-indexed, but the
entity/fact counts used for context are.

**If the two graphs are identical, the toggle is inert, the comparison is VOID, and
no conclusion may be drawn from the scores.** That is the failure mode that has
produced false "measured null" findings in this project before.

## Noise floor

n = 199. For an unpaired proportion near 0.8, 1 SE ≈ 2.8pt, so 2 SE ≈ 5.7pt.
The design is **paired**, so the SE on the *difference* is smaller — it depends on
the discordant-pair count, not the base rate. With moderate discordance (~20 of 199
flipping) the paired 2 SE is ≈ 4.5pt.

**A single-run difference below ~5pt is not distinguishable from noise.**

## THE DECISION RULE

Let `Δ = qa-gen(dream-on) − qa-gen(dream-off)`, in percentage points.

| condition | outcome |
|---|---|
| validity gate fails (graphs identical) | **CUT** — unanswerable in one session |
| either arm fails to complete | **CUT** — unanswerable in one session |
| `Δ < +5.0` | **CUT** |
| `Δ ≥ +5.0` | **KEEP (provisional)** — and only if the runtime cost is stated alongside |

Reported as supporting evidence, decisive of nothing: McNemar exact p on the
discordant pairs, and the discordance count itself (which independently shows
whether the lever moved any answer at all).

## Accepted limitation, stated in advance

n = 1 run per arm. This design **cannot** detect a small effect, and it is not
trying to. The bar is set deliberately high because a small effect would not
justify dream's cost anyway: ~90 minutes of wall-clock and a demonstrated ability
to destroy entities. A lever that needs multiple runs to distinguish from zero is,
for this decision, indistinguishable from zero.

---

# OUTCOME (appended 2026-08-19, after measuring)

## Validity gate: **PASSED**

`dream-on` 10 live `entity_merge` rows; `dream-off` 0. Retrieval differed on
**199/199** questions, mean Jaccard 0.338. The lever was unambiguously active.

## The primary metric was NOT run — and it did not need to be

qa-gen requires ~$1 of OpenAI spend (`qa_eval.py` hardcodes `api.openai.com`;
answerer `gpt-4o-mini`, judge `gpt-4o`), which was not authorised. The pre-registered
rule therefore returns **NO VERDICT** on its own terms, and that is recorded honestly.

**Substring was correctly declared non-decisive in advance, and this is why:** it
scored **148/152 = 97.4% in BOTH arms**, i.e. it sits at its ceiling. Maximum
possible gain is +2.6pp, so the +5.0pp bar was arithmetically unreachable there. A
"CUT on substring" would have been an artefact of a saturated scorer. **The
pre-registration prevented exactly the wrong answer it was written to prevent.**

## What decided it instead: `evidence_eval.py` — free, offline, instrument-validated

Validated BEFORE use via its own `--self-test`: shuffling retrieved lists moves
nDCG 64.0% → 10.7% and recall 77.2% → 21.9%. Rank- and set-sensitive, with real
headroom, measuring the retrieval layer dream actually acts on.

### The 2×2 (n=149 scorable, k=10, four local runs, £0)

| `L4_LEXICAL_JACCARD_MIN` | dream OFF | dream ON | dream's cost |
|---|---|---|---|
| **0.5** (shipped) | 64.0 nDCG | 54.1 | **−9.9** |
| **0.6** (experiment) | 63.8 | 63.4 | **−0.4** |

## Root cause, proven

All 10 merges were `site=canonicalize`, `struct=false`, `cos=None` — **L5
surface-form canonicalisation**, not the acronym pass (already default-off) and not
any LLM-adjudicated site. `find_merge_pairs` selects on **cosine alone**; cosine
between short noun phrases measures topical relatedness, not identity.

Every merge was a hypernym collapse destroying the distinguishing token:
`lgbtq community`→`community`, `pottery class`→`pottery`, `pottery
workshop`→`pottery`, `transgender poetry reading`→`poetry reading`, and six more.

**8 of the 10 sat at Jaccard EXACTLY 0.500** — the signature of "one name is the
other plus one token", i.e. specialisation.

## Why the threshold fix was REVERTED despite working

Raising 0.5 → 0.6 cut dream's harm from −9.9 to −0.4 and took corpus precision
0.9545 → 1.000. It was still the wrong layer:

- `alice j` → `alice johnson` and `Ria Patel` → `Ria` are **also J=0.500**.
  Initial-abbreviated person names are the same shape as `pottery class`/`pottery`.
- **No threshold separates them.** 8 tests across 5 files encode the abbreviation
  case as core; corpus overall recall fell 0.568 → 0.263.
- Separating them needs an initialism check — a fourth per-pair discriminator,
  banned by decision `20e1f4e3` after three failed attempts.

## The actual fix (next session)

**L5 canonicalisation is the ONE identity site ADR-063 never gave an adjudicator.**
Of the six sites, #3 and #5 route candidates through LLM adjudication → `write_gate`;
#6 touches `canonicalization.rs` but only combines confidence (noisy-OR), it does not
decide pairs. The impl-spec treats L5 as *precedent*, stating Site #5 differs from it
in exactly "CANDIDATE GENERATION + ADJUDICATION".

An LLM separates `alice j`/`alice johnson` from `pottery class`/`pottery` trivially;
token-Jaccard provably cannot. `lexical.rs:73-79` already documents this routing for
the `Amazon`/`Amazon River` false positive — L5 was simply never wired to it.

Shape: keep the lexical gate at **0.5 as a NOMINATOR**, demote it from decider, and
route survivors through the existing `write_gate` + `IdentityVerdictBatch` machinery.
Not new architecture — the ratified pattern applied to the site that was skipped.
`write_gate` Row 6 still holds: the LLM never authorises a destructive write alone.
Cost is bounded — only pairs passing the cheap gate are adjudicated (10 on conv0),
and dream is latency-tolerant by design.

## Honest limitations

- **n = 1 per cell**, and the four arms were **independently ingested**, so extraction
  stochasticity is confounded with the levers. Entity counts historically span
  120–215 across runs. A cleaner design ingests once, copies the DB, and dreams on
  the copy — that isolates dream, though not the L4 threshold, which affects ingest.
- The direction was consistent across 4 aggregate measures and 5 categories, which is
  not what noise usually looks like — but it is not excluded.
- `run_local_ingest.sh` **misreports** its own summary: it printed `entities final
  (post-dream): 189` / `dream merge delta: 0` when the DB held 179 entities and 10
  merges. Its trajectory sampler is killed by the exit trap before dream's merges
  land. Anyone reading that log concludes dream merged nothing. ~~NOT FIXED.~~
  **FIXED 2026-08-19 (`8d38d20a`)** — `FINAL` now reads the DB, and the merge count
  comes straight from `graph_mutation_log`. The root cause was slightly different
  from the guess recorded above: the sampler is not killed early, it simply samples
  every 60s and dream's merges land inside that last window. Both queries were
  validated against the two known-truth DBs before the fix was committed
  (`dream-off` → 0 merges, `dream-on` → 10, entities 187 / 179).

## What is explicitly OUT of scope for this session

No defect fixing found along the way unless it blocks the run. Anything else gets
one line in the register and nothing more. The failure mode being guarded against
is the one that produced this decision.

---

# PRE-REGISTRATION 2 — the adjudicator run (written BEFORE measuring)

Added 2026-08-19, after `2fd9a48e` wired L5 through the ADR-063 `write_gate` and
BEFORE any re-measurement. Same discipline as the first pre-registration, for the
same reason: afterwards I will reinterpret whatever I find to fit the conclusion I
already hold.

## The vacuity trap this exists to catch

The obvious success criterion — **"dream-on nDCG within noise of dream-off"** — is
satisfied *perfectly* by a completely broken adjudicator.

The gate is deliberately fail-closed: no verdict ⇒ `Reject` ⇒ no merge. So if the
local Ollama model returns garbage, times out, or simply answers `false` to
everything, **dream merges nothing at all**, dream-on becomes dream-off by
construction, and the headline number looks like a total success. I would have
shipped an inert pass and recorded it as a fix.

A null result here is therefore only meaningful alongside evidence that the
adjudicator was *awake*.

## The pre-registered conditions

A **PASS** requires ALL THREE. Any one failing is not a partial pass:

1. **Recovery** — dream-on nDCG@10 ≥ dream-off − 2.0 (was −9.9). Same
   `evidence_eval`, same instrument-validation self-test run first.
2. **NON-VACUITY** — the adjudicator both merged AND rejected. Concretely, in
   `identity_verdict_audit` on the dream-on DB:
   `count(decision='merge') ≥ 1` **AND** `count(decision='reject') ≥ 1`.
   All-reject ⇒ the pass is inert and condition 1 is meaningless.
   All-merge ⇒ the adjudicator is a rubber stamp and nothing was fixed.
3. **The abbreviation case survives** — no `alice j`-shaped merge is lost. Checked
   against the dream-on DB's audit rows, not asserted from the unit tests.

## What would REVERSE this and what I would do about it

| observation | verdict | action |
|---|---|---|
| 1 ✓, 2 all-reject | **NOT a fix** — an inert pass wearing a green number | local model too weak for adjudication; the fix is real but unmeasurable on this model. Escalate the model, do NOT record a pass |
| 1 ✓, 2 all-merge | **NOT a fix** — rubber stamp | the prompt or the parse is wrong; inspect `reasoning` text before anything else |
| 1 ✗ (still ≤ −2.0) | adjudicator insufficient | read the audit `reasoning` rows for the merges that survived — do NOT reach for a threshold |
| 3 ✗ | regression | the fix traded one failure for the other; that is the reverted threshold change by another route |

If cost 1 recovers but I cannot show 2, the honest report is **"no verdict"**, exactly
as the first pre-registration returned no verdict on qa-gen.

## Still unaddressed, and stated so it is not quietly dropped

- **qa-gen remains the PRIMARY metric and is still unrun** (~£1 OpenAI, unauthorised).
  `evidence_eval` is the secondary this rests on. The formal rule still returns NO
  VERDICT on its own terms.
- **n = 1 per cell, independently ingested** — unchanged from above, and now with one
  extra source of variance: the adjudicating LLM's own non-determinism.
- Wall-clock is **~1.5h per arm** per the script's own header (the session note's
  "~25 min" is wrong), so a 2×2 is ~3h.

---

# OUTCOME 2 (appended 2026-08-20, after measuring the adjudicated arm)

Run `dream-on-adj`, release binary rebuilt from `2fd9a48e` (a hard precondition in the
launcher refused to start on the stale 18:25 binary, which predated the fix by 3h and
would have measured the OLD code). Local Ollama, money guard PASSED, £0. Wall-clock
**24 min**, not the ~1.5h the script's header claims — that header is stale.

## Condition 2 — NON-VACUITY: **PASS** (checked FIRST, before looking at any score)

`identity_verdict_audit`, site `l5_canonicalize`: **1 merge, 6 rejects.** The adjudicator
was awake and discriminating — neither inert nor a rubber stamp. Canonicalize merges fell
**10 → 1** in `graph_mutation_log`.

This was checked before the score deliberately: the fail-closed gate makes "dream-on ==
dream-off" the signature of a *broken* adjudicator as well as a working one.

## Condition 1 — RECOVERY: **PASS**

`evidence_eval` self-test run first (shuffle moves nDCG 64.1 → 10.9, recall 77.9 → 22.7).

| arm | recall@10 | **nDCG@10** | MRR | hit-rate |
|---|---|---|---|---|
| dream OFF (baseline) | 77.2% | **64.0%** | 58.3% | 83.2% |
| dream ON — unadjudicated | 75.2% | **54.1%** | 45.8% | 81.2% |
| **dream ON — adjudicated** | **77.9%** | **64.1%** | **58.4%** | **83.9%** |

**The −9.9 nDCG is fully recovered: 54.1 → 64.1, +10.0.** All four measures move together
and all three now sit at or fractionally above the dream-off baseline. Bar was ≥ 62.0.

## Condition 3 — the abbreviation case: **NOT EXERCISED. Not passed.**

Stated plainly because the distinction matters: **conv0 does not contain the `alice j`
shape at all.** All 10 merges in the unadjudicated run were hypernym/category collapses
(`pottery class`→`pottery`, `lgbtq community`→`community`, `those kids`→`kids`, …), and all
7 nominations in this run were too. There was no abbreviated person name to preserve or
lose, so this run neither confirms nor refutes the condition. The only evidence for it
remains the fast-tier unit tests plus one scripted mock — **no real model has been shown to
make that call.** The adversarial VCR fixture (TD-225 DoD (c)) is what would close it.

## What the adjudicator actually decided — the whole audit trail

**MERGED (1):** `stained glass window` → `staind glass window`, conf 1.0 — *"The difference
is a common spelling variation."* Correct.

**REJECTED (6),** every one the exact class that cost 9.9 nDCG:

| loser → keeper | conf | the model's reasoning |
|---|---|---|
| `pottery class` → `pottery` | 0.95 | "'Pottery' is the subject matter, while 'pottery class' refers to an activity about that subject." |
| `family time` → `family` | 0.90 | "The noun 'family' is a group, while 'family time' refers to a specific activity or period." |
| `exploring nature` → `nature` | 0.90 | "'Nature' is the topic, whereas 'exploring nature' describes an activity related to it." |
| `18th birthday` → `18th` | 0.90 | "'18th' denotes a numerical descriptor, while '18th birthday' denotes a specific event." |
| `adoption` → `adoption agencies` | 0.90 | "An agency is a place/organization, while 'adoption' is the concept or process." |
| `selfacceptance and support` → `selfacceptance and finding support` | 0.80 | "Dropping 'finding' changes the scope of support being offered." |

## The honest reading: the HARM is removed; a BENEFIT was never demonstrated

`+0.1` nDCG over dream-off is **inside the measured noise floor** — two independent
dream-off ingests scored 64.0 and 63.8, i.e. ±0.2. So the correct claim is that **dream is
now NEUTRAL for retrieval**, not that it helps.

That leaves the ORIGINAL keep-or-cut question exactly where it was: dream now costs ~24 min
of wall-clock for no measurable retrieval gain. What changed is that it is no longer
actively destroying 9.9 nDCG of quality, which is the precondition for asking whether its
other outputs are worth the runtime — not an answer to it.

## Limitations, unchanged and stated

- **Independently ingested, n = 1 per cell.** Entity counts: 187 (off) / 179 (on) / 189
  (adj). The candidate sets therefore DIFFER between arms — 7 pairs nominated here vs 10
  merged before — so this is **not** a paired comparison of the same pairs, and extraction
  stochasticity is still confounded with the lever. The paired-DB design (ingest once, copy,
  dream on the copy) remains the clean version and remains unbuilt.
- 2 arm failures this run (TD-200 confound control), in line with prior runs.

---

# OUTCOME 3 (appended 2026-09-02, after running qa-gen — THE FORMAL VERDICT)

Run against the same `dream-off.db` / `dream-on-adj.db` from OUTCOME 2, no re-ingest, no
re-query — the retrieval-result JSON files (`dream-off.json`, `dream-on-adj.json`) were fed
straight into the pre-registered `qa_eval.py` pipeline (`answer-gen` → `answer-judge` →
`answer-tally`). Local Ollama for retrieval, OpenAI `gpt-4o-mini` (answerer) + `gpt-4o`
(judge) for scoring, per the original design. Cost: **$0.373** ($0.136 dream-off answer-gen
+ $0.237 dream-off answer-judge; dream-on-adj hit a pre-existing cache from an incomplete
2026-08-20 attempt at $0.000). Key sourced from an unrelated sibling project's `.env`
(`kgentic-contextify-autodocs`), used with explicit authorisation.

## Validity gate: **PASSED** (re-verified, and the first attempt at this was itself wrong)

`graph_mutation_log` row counts, via plain `COUNT(*)` (this table is NOT vector-indexed, so
the `recorded_at`-line-counting workaround for the `entities`/`facts` gotcha does not apply
here — confirmed by schema read after an initial `SELECT recorded_at FROM
graph_mutation_log` errored on a nonexistent column and the resulting error text was
miscounted as "3 rows" by a blind `wc -l`, in both arms, before the actual query output was
inspected): **dream-off = 0, dream-on-adj = 1.** The one row is the same `stained glass
window` → `staind glass window` merge OUTCOME 2 already found. Lever was genuinely active,
not an inert toggle.

## Primary metric — qa-gen, n=152 paired questions

| arm | correct | total | accuracy |
|---|---|---|---|
| dream OFF | 124 | 152 | **81.6%** |
| dream ON — adjudicated | 122 | 152 | **80.3%** |

**Δ = 80.3 − 81.6 = −1.3pp.**

Paired McNemar (question IDs matched 152/152 across both independently-ingested arms — the
LoCoMo question set itself is stable even though retrieval and candidate entities are not):
**8 discordant pairs where dream-off was right and dream-on-adj wrong, 6 the other way, 14
discordant total, exact p = 0.79.** Not significant; consistent with pure noise around zero.

## THE VERDICT, per the decision rule pre-registered above

Validity gate passed. Both arms completed (152/152, 0 failures in either stage).
**Δ = −1.3pp < +5.0pp → CUT.**

This is not "no verdict" any more. It is the first time in this project's history that the
formally pre-registered primary metric for dream has actually run, and it says: **dream's
harm is fixed (OUTCOME 2), and dream provides no measurable benefit on the metric that was
declared decisive in advance** — slightly negative, not distinguishable from zero, ~24
minutes of wall-clock either way.

## What this does and does not settle

- **Settles**: whether dream, as currently built (all 5+ passes, L5 now adjudicated),
  clears the bar this project itself set for keeping it on `conv0`. It does not.
- **Does not settle**: whether some SUBSET of dream's passes (e.g. type discovery, alias
  resolution) independently earns its keep outside the aggregate — this measurement is only
  of the whole `mem.dream()` call.
- **Does not settle**: what to actually DO about the default (turn dream off by default,
  make it explicitly opt-in, leave as-is and just record the finding) — that is a product
  decision, not a measurement one, and is deliberately left open here for the project owner.
