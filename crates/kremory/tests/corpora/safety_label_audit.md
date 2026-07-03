---
title: Cross-family label audit — safety-category rows (Site #3 / Site #5 adversarial corpora)
type: audit
status: complete
created: 2026-07-03
source: local Ollama qwen2.5:7b (independent re-labeling, cross-family from Claude Sonnet corpus generation)
---

# Cross-Family Label Audit — Safety-Category Rows

## Why this audit exists

`site5_acronym_adversarial.jsonl` and `site3_type_collapse_adversarial.jsonl` were authored by
Claude (Sonnet). Per `~/.claude/rules/research.md`'s model-family independence rule, Claude
scoring/labeling its own generated content is **not independent verification** — same-family
RLHF correlation can surface shared bias as apparent agreement rather than ground truth. The
"must not merge" (safety) rows are the load-bearing labels for the zero-false-merge gate (ADR-063
spec §5.2 / the Site #3 and Site #5 spikes), so they were re-labeled independently using
**qwen2.5:7b** (Alibaba Qwen family — different lineage from Claude) via local Ollama.

## Scope audited

83 rows total, across the 5 categories that assert "these are DIFFERENT entities/concepts — do
not merge":

| Corpus | Category | Row count |
|---|---|---|
| `site5_acronym_adversarial.jsonl` | `coincidental_collision_distinct` | 15 |
| `site5_acronym_adversarial.jsonl` | `distinct_people_same_nickname` | 13 |
| `site3_type_collapse_adversarial.jsonl` | `distinct_lemma_collision` | 22 |
| `site3_type_collapse_adversarial.jsonl` | `distinct_unrelated` | 15 |
| `site3_type_collapse_adversarial.jsonl` | `band_edge_moderate` | 18 |
| **Total** | | **83** |

## Method

1. Extracted the 83 safety-category rows from both JSONL corpora.
2. Smoke-tested ONE row (`s5-028`, ABC News / Acme Business Consulting) against qwen2.5:7b via the
   Ollama HTTP API (`/api/generate`, `temperature=0.0`) before batching — response was clean and
   correctly "DIFFERENT". Proceeded to the full batch per `smoke-one-before-batch-llm-validation`.
3. For Site #5 rows: prompted qwen with `Are '<a>' and '<b>' the same real-world entity? Answer
   SAME or DIFFERENT on the first line, then a short reason.` — no mention of kremory, ADR-063, or
   the existing corpus label (independence preserved).
4. For Site #3 rows: prompted qwen with `Do the type names '<a>' (<desc_a>) and '<b>' (<desc_b>)
   denote the same concept? Answer SAME or DIFFERENT + reason.`
5. Mapped qwen's verdict against the corpus's `same_entity` / `same_concept` field (both `false`
   for every row in scope — these are all "must stay distinct" rows). Agreement = qwen said
   DIFFERENT.
6. Computed agreement rate; investigated every disagreement individually (see below).

## Result

**Agreement rate: 80 / 83 = 96.4%** — PASS (≥ 90% bar per spec §5.2).

| Corpus | Rows | Agreements | Disagreements |
|---|---|---|---|
| Site #5 (28 rows: 15 collision + 13 nickname) | 28 | 25 | 3 |
| Site #3 (55 rows: 22 lemma + 15 unrelated + 18 band-edge) | 55 | 55 | 0 |
| **Total** | **83** | **80** | **3** |

Site #3's three safety categories hit **100% agreement** (55/55) — qwen2.5:7b independently
confirmed every `distinct_lemma_collision`, `distinct_unrelated`, and `band_edge_moderate` row as
genuinely distinct concepts, with zero false "SAME" verdicts on the false-merge-risk rows.

## Disagreements — full detail + judgment

All 3 disagreements are in Site #5's `coincidental_collision_distinct` category:

| id | a | b | corpus label | qwen verdict (bare prompt) |
|---|---|---|---|---|
| s5-033 | MAS | Monetary Authority of Singapore | distinct (different entity) | **SAME** — "MAS is an abbreviation for Monetary Authority of Singapore" |
| s5-038 | BA | Bachelor of Arts | distinct (different entity) | **SAME** — "Both terms refer to an undergraduate degree in arts disciplines" |
| s5-041 | PSA | Prostate-Specific Antigen | distinct (different entity) | **SAME** — "PSA is the abbreviation for Prostate-Specific Antigen" |

### Root cause investigation

These 3 rows are the only ones (of 15 `coincidental_collision_distinct` rows) where the `a` field
is a **bare 2-3 letter acronym with zero disambiguating context** AND the acronym's single most
globally-dominant expansion happens to coincide with `b`. The corpus's own `rationale` field for
these rows makes clear the *intended* reading of `a` is a **different, specific expansion** than
`b` (e.g. "MAS in one context refers to Malaysia Airlines System... while in another it is
Singapore's central bank"; "BA as British Airways and BA as Bachelor of Arts... unrelated
entities"; "PSA as Public Service Announcement and PSA as Prostate-Specific Antigen... different
referents"). My initial audit prompt supplied qwen with only the bare acronym string for `a` — no
context to select the intended alternate reading — so qwen defaulted to the single most salient
global sense of the acronym, which for these three specific acronyms happens to match `b`.

This is corroborated by the real system's own design: the S2 spike fixture
(`crates/kremory/tests/acronym_nickname_recall_s2_spike.rs`) that actually exercises the
`write_gate` LLM adjudication for this category **always supplies both entities' descriptions**
(`a_desc` / `b_desc` fields) to the adjudicating LLM — it never asks the model to resolve a bare,
undisambiguated acronym in isolation. The other 12 of 15 collision rows in the corpus (SUN, AA,
PAN, CVS, IRA, ADA, CIA, MADD, NAFTA, ABC News, WHO (the rock band), ACE Hardware) all correctly
scored DIFFERENT under the same bare-string prompt, because those `a` values already carry enough
inherent salience-splitting signal (either the acronym's dominant sense differs from `b`, or the
`a` field itself includes disambiguating text like "(the rock band)").

**Verification**: re-running the 3 disputed rows with disambiguating context lifted directly from
each row's own `rationale` field (e.g. "Entity A is 'MAS', referred to in a regional business
report about the old Malaysia Airlines System. Entity B is 'Monetary Authority of Singapore', the
central bank of Singapore. Are Entity A and Entity B the same real-world entity?") flipped qwen's
verdict to **DIFFERENT** on all 3, unanimously agreeing with the corpus label:

- MAS: "MAS in this context refers to the former Malaysia Airlines System... The Monetary
  Authority of Singapore (MAS) is the central bank... distinct entities."
- BA: "'BA' as an airline carrier ... 'Bachelor of Arts' as an undergraduate degree ... entirely
  different concepts."
- PSA: "PSA in the context of a Public Service Announcement ... PSA (Prostate-Specific Antigen)
  ... distinct entities with different meanings and applications."

### Judgment

**All 3 disagreements are audit-prompt artifacts, not corpus label errors.** The corpus labels for
`s5-033`, `s5-038`, `s5-041` are correct as authored. The disagreement arose because my
independent-audit prompt (necessarily, to preserve independence — no corpus label or rationale was
shown to qwen) stripped context that the real production adjudication pipeline always supplies.
When given the same disambiguating context the corpus's own rationale specifies (and which the
real S2 pipeline supplies via entity descriptions), qwen unanimously agrees with the corpus.

No corpus edits are needed. This is noted as an **audit-methodology finding**, not a data-quality
finding: bare-acronym-only prompting is a weaker proxy for these 3 specific high-global-salience
acronyms (MAS, BA, PSA) than for the corpus's other 12 collision rows — a limitation of the
label-audit method, not of the corpus.

## Verdict

**PASS** — 96.4% agreement (80/83) exceeds the 90% bar per spec §5.2. Zero of the 3 disagreements
reflect an actual corpus mislabel; all 3 are resolved as audit-prompt-context artifacts upon
supplying the same disambiguating context the real adjudication pipeline and the corpus's own
rationale field already carry. Both Site #3 safety categories (55 rows, 3 categories) hit 100%
independent agreement with zero exceptions.

**No corpus corrections required.**
