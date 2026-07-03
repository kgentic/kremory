# Site #5 Acronym Adversarial Corpus — v2 Changelog

**File**: `crates/kremory/tests/corpora/site5_acronym_adversarial.jsonl`
**v1 row count**: 121 (s5-001 .. s5-121)
**v2 row count**: 140 (s5-001 .. s5-141, with s5-131 intentionally omitted — see "Deliberate omission" below)

## Governing rule applied throughout

Labels encode **verifiable real-world fact** plus the **provided context**, decided independent of what any LLM would answer. A label may be **corrected** when it is factually wrong. A label may **never** be flipped just because a model merged or rejected the pair. Every change below carries its factual justification. Where a second referent could not be verified as genuinely real, the row was corrected or dropped rather than guessed — per "if unsure whether two things are the same entity, keep it OUT of the safety categories."

---

## The problem that motivated v2

The v1 `coincidental_collision_distinct` category systematically mispaired several rows: it took an acronym (`a`) and paired it with **its own genuine expansion** (`b`), then labeled the pair `same_entity: false` under the category name "coincidental collision distinct." But an acronym and its true expansion **are the same entity** — that is not a collision, it is correct identity. A true acronym collision requires **the same acronym string mapping to two different, independently real expansions**, both named explicitly enough to be unambiguous.

Affected v1 rows: s5-031 (AA), s5-033 (MAS), s5-034 (PAN), s5-035 (CVS), s5-036 (IRA), s5-037 (ADA), s5-038 (BA), s5-039 (CIA), s5-040 (MADD), s5-041 (PSA), and s5-085 (NAFTA).

---

## TASK 1 — Relabeled rows (factual corrections)

For each row below, `a` was verified to be the genuine, standard acronym/abbreviation of `b`. Both refer to one real-world entity. Category changed from `coincidental_collision_distinct` / `same_entity: false` → `genuine_acronym` / `same_entity: true`.

| ID | a | b | Factual verification |
|---|---|---|---|
| s5-031 | AA | Automobile Association | AA is the standard abbreviation of the UK's Automobile Association, a real roadside-assistance/motoring membership organization. Same entity. |
| s5-033 | MAS | Monetary Authority of Singapore | MAS is the official acronym of Singapore's central bank/financial regulator. Same entity. |
| s5-034 | PAN | Personal Area Network | PAN is the standard networking-industry acronym for Personal Area Network. Same concept. |
| s5-035 | CVS | Concurrent Versions System | CVS is the standard acronym for the Concurrent Versions System version-control tool. Same entity. |
| s5-036 | IRA | Irish Republican Army | IRA is the standard acronym for the Irish Republican Army. Same entity. |
| s5-037 | ADA | Americans with Disabilities Act | ADA is the standard acronym for the US Americans with Disabilities Act. Same entity. |
| s5-038 | BA | Bachelor of Arts | BA is the standard abbreviation for the Bachelor of Arts degree. Same entity/concept. |
| s5-039 | CIA | Culinary Institute of America | CIA is the acronym used by the (real) Culinary Institute of America. Same entity. |
| s5-041 | PSA | Prostate-Specific Antigen | PSA is the standard medical acronym for Prostate-Specific Antigen. Same entity/concept. |

### s5-040 (MADD) — special case, corrected expansion rather than simple relabel

v1 paired MADD with **"Maryland Association of Dental Distributors"** — a fabricated/fictional organization with no verifiable real-world existence (the v1 rationale even called it "a fictional regional dental distributors trade group"). This cannot be corrected by a simple relabel because the `b` field itself is not real. Per the anti-massaging rule, a row cannot claim `same_entity` (or, symmetrically, a collision) against an entity that doesn't exist. **Fix applied**: `b` replaced with MADD's actual, verifiable expansion — **Mothers Against Drunk Driving** (founded 1980, real US nonprofit). Category → `genuine_acronym`, `same_entity: true`. No second real, independently-notable MADD expansion could be verified with acceptable confidence, so MADD was **not** used to build a true-collision row (see Task 2 note below).

### s5-085 (NAFTA) — same fabrication pattern, same fix

v1 paired NAFTA with **"National Aeronautics Foundation of Texas Association"** — also fabricated, no real-world existence. **Fix applied**: `b` replaced with NAFTA's actual, verifiable expansion — **North American Free Trade Agreement**. Category → `genuine_acronym`, `same_entity: true`. No verifiable second real-world NAFTA expansion exists, so NAFTA was not used to build a true-collision row either.

**Net effect of Task 1**: 11 rows corrected (9 simple relabels + 2 fabricated-expansion fixes). `genuine_acronym` category grows from 30 (v1) to 41 (v2).

---

## TASK 2 — Kept-distinct rows + true collisions built

### Kept as-is (genuinely distinct — the name itself disambiguates)

- **s5-028** (ABC News vs Acme Business Consulting) — kept. Different real organizations; only the bare initialism "ABC" collides, not fuller names.
- **s5-029** (WHO the rock band, i.e. The Who, vs World Health Organization) — kept, with `a` field clarified to "WHO (the rock band, The Who)" for precision. Different real entities.
- **s5-032** (ACE Hardware vs ACE Advanced Computing Environment) — kept. Different real organizations.

Also kept unchanged (not named in the task list but same shape, verified correct on inspection): **s5-030** (SUN the star vs Sun Microsystems/Stanford University Network — `a` clarified to "SUN (the star)" and `b` clarified to "Sun Microsystems (originally Stanford University Network)" for precision; both real, distinct referents).

### TRUE collisions built (same acronym, two different real expansions, both named) — 9 new rows, s5-122 through s5-130

Each pair below was verified: both expansions are real, independently notable, and share the identical acronym string.

| ID | Acronym | Expansion 1 | Expansion 2 |
|---|---|---|---|
| s5-122 | MAS | Monetary Authority of Singapore (Singapore's central bank) | Malaysia Airlines System (former legal name of the Malaysian flag carrier) |
| s5-123 | PAN | Personal Area Network (networking concept) | Permanent Account Number (India's taxpayer ID, issued by the Income Tax Department) |
| s5-124 | CVS | Concurrent Versions System (legacy version-control tool) | CVS Pharmacy / CVS Health (US retail pharmacy chain) |
| s5-125 | IRA | Individual Retirement Account (US retirement savings vehicle) | Irish Republican Army (paramilitary organization) |
| s5-126 | ADA | American Diabetes Association (US nonprofit) | Americans with Disabilities Act (US civil rights law) |
| s5-127 | BA | British Airways (UK flag carrier) | Bachelor of Arts (academic degree) |
| s5-128 | CIA | Central Intelligence Agency (US federal intelligence agency) | Culinary Institute of America (culinary arts college) |
| s5-129 | PSA | Public Service Announcement (media messaging form) | Prostate-Specific Antigen (medical blood test marker) |
| s5-130 | AA | Alcoholics Anonymous (recovery fellowship) | Automobile Association (UK motoring membership org) |

All 9 rows: `category: coincidental_collision_distinct`, `same_entity: false`, both expansions explicitly named in the `a`/`b` fields so the distinction is unambiguous even without extra context.

Combined with the 4 kept rows (s5-028, s5-029, s5-030, s5-032), the corpus now has **13 rows** in `coincidental_collision_distinct` — all genuinely distinct, two-real-referent collisions.

### Deliberate omission — MADD true collision NOT built

An initial draft attempted to pair MADD (Mothers Against Drunk Driving) with a second "real" expansion to complete the collision set to match the 10 acronyms corrected in Task 1. No second MADD expansion could be verified as genuinely real and independently notable — candidate expansions considered (e.g. a vendor product-line name) could not be confirmed as real. Per the anti-massaging rule ("if unsure whether two things are the same entity, keep it OUT of the safety categories" — which extends symmetrically to "don't invent a second entity to force a collision"), **no MADD collision row was added**. A placeholder row was drafted at id `s5-131` during editing and then **deleted** rather than shipped, to avoid leaving a fabricated entity in the corpus. This is why v2's ID sequence has a gap at 131 — intentional, not a data-entry error.

NAFTA was excluded from the true-collision set for the same reason.

---

## TASK 3 — `context_facts` field added to every row

A new schema field `context_facts` (list of 1–3 short strings) was added to **all 140 rows** in the corpus — both the 121 pre-existing rows (backfilled from their existing `rationale` field) and the 19 new rows (10 collision-related + already covered, 10 nickname rows — see Task 4). `context_facts` states the concrete, plantable discriminating facts a real transcript/ingest pipeline would supply: roles, relationships, IDs, affiliations, shared record numbers, or explicit "no relation" statements.

- For `same_entity: true` rows, `context_facts` establishes **sameness** (e.g. shared surname, shared employee ID, explicit tie in one profile).
- For `same_entity: false` rows, `context_facts` establishes **distinctness** (e.g. different departments, different IDs, explicit "no relation," different real organizations).
- For `same_entity: uncertain` rows, `context_facts` describes **why the context is genuinely too thin** to disambiguate.

Verified: `python3` JSONL parse confirms all 140 rows contain a non-empty `context_facts` array.

---

## TASK 4 — `context_dependent_nickname` category (10 new rows, s5-132 .. s5-141, 5 matched pairs)

New category testing whether context — not name pattern alone — drives the merge decision. Each pair uses the **identical name/nickname pair** twice: once with context proving sameness (`same_entity: true`), once with context proving distinctness (`same_entity: false`). This directly probes whether an LLM defaults to a name-similarity heuristic or actually incorporates supplied facts.

| Pair | Name pattern | Same-entity row (context) | Distinct row (context) |
|---|---|---|---|
| 1 | Pat / Patricia Nolan / Patrick Nolan | s5-132: same employee ID CS-4471, single rep on the call, transcription variance | s5-133: siblings, different departments, different employee IDs |
| 2 | Frankie / Frank Costa / Francesca Costa | s5-134: same dial-in extension x4471, single "Costa" attendee, transcription variance | s5-135: siblings, separate meetings/employee numbers |
| 3 | Sam / Samuel Whitlock / Samantha Whitlock | s5-136: same support agent ID AGT-2291, one agent on the ticket | s5-137: different agent IDs, roster confirms never same shift |
| 4 | Alex / Alex Whitmore (night) / Alex Whitmore (day) | s5-138: same payroll ID PR-5521, one active HR record | s5-139: different payroll IDs, HR flags "unrelated, same name" |
| 5 | Jo / Joanna Reyes / Joseph Reyes | s5-140: same account-manager ID AM-8834, single contract signer | s5-141: married couple, separate account-manager IDs, non-overlapping account sets |

Note: pair 1 (s5-132/s5-133) intentionally reuses the exact name pair from the pre-existing v1 row s5-051 (Pat/Patricia/Patrick, `distinct_people_same_nickname`, `same_entity: false`, unconditionally). s5-051 remains unchanged as the "context-free default" baseline; s5-132/s5-133 are new, separate rows that show the SAME name pair flips to `true` or stays `false` depending purely on supplied context — the direct A/B contrast the task asked for.

---

## Verification summary

Ran a Python JSONL-parse validation pass:

- 140/140 rows parse as valid JSON, one per line.
- 140/140 unique `id` values (gap at `s5-131` is intentional — see "Deliberate omission").
- 140/140 rows have a non-empty `context_facts` array.
- Category counts (v2): `genuine_acronym` 41, `genuine_nickname_cooccurring` 30, `coincidental_collision_distinct` 13, `distinct_people_same_nickname` 13, `uncertain_thin_context` 8, `non_cooccurring_nickname` 11, `hard_negatives_unrelated` 8, `unicode_casing_variant` 6, `context_dependent_nickname` 10 (new).
- `same_entity` distribution (v2): `true` 93, `false` 39, `"uncertain"` 8.

## Rows I was unsure about + how resolved conservatively

1. **s5-040 (MADD)** and **s5-085 (NAFTA)** — v1's paired "expansion" was fabricated in both cases. Rather than guess a plausible-sounding but unverifiable second real expansion to preserve a "collision" example, I corrected both to their real expansion (relabeled to `genuine_acronym`) and explicitly declined to manufacture a second collision partner. Resolved conservatively per the rule: don't guess a fact just to fill a category slot.
2. **s5-029 (WHO) and s5-030 (SUN)** — the original `a` fields ("WHO (the rock band)" / "SUN") were slightly ambiguous about which real-world referent was meant. I clarified the `a`/`b` text (e.g. "WHO (the rock band, The Who)", "SUN (the star)" / "Sun Microsystems (originally Stanford University Network)") without changing the label or category — purely a precision edit so the row is unambiguous to a human reader, not a factual correction.
3. **New true-collision candidates beyond the 9 built** — I considered adding a 10th collision (e.g. for MADD or NAFTA) to round the count to an even dozen, but declined per point 1 above. 13 total collision rows (4 kept + 9 new) was judged sufficient coverage without forcing an unverifiable pairing.
