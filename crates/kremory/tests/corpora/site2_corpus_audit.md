---
title: Cross-family label audit — Site #2 type-novelty adversarial corpus (ADR-065 Phase 0)
type: audit
status: complete
created: 2026-07-03
source: local Ollama qwen2.5:7b (independent re-labeling, cross-family from Claude Sonnet corpus generation)
refs:
  - adr-065-site2-type-novelty-decouple-write-gate-2026-07-03
  - adr-065-site2-fix-impl-spec-2026-07-03
raw_data: crates/kremory/tests/corpora/site2_audit_qwen_raw.json
---

# Cross-Family Label Audit — Site #2 Type-Novelty Corpus

## Why this audit exists

`site2_type_novelty_adversarial.jsonl` (26 rows) was authored by Claude (Sonnet). Per
`~/.claude/rules/research.md`'s model-family independence rule, Claude labeling its own generated
content is **not** independent verification — same-family RLHF correlation surfaces shared bias as
apparent agreement. ADR-065 Finding 1 (HIGH) requires this corpus to clear an independent
cross-family audit **before** Site #2 is enabled, because the harness gate validates the fix
*against these labels* — bad ground truth would make a green gate meaningless (Critical Rule 10).

Independent second labeler: **qwen2.5:7b** (Alibaba Qwen family — distinct RLHF lineage from
Claude), via local Ollama. Same model + methodology family as the Site #3/#5 safety audit
(`safety_label_audit.md`, qwen2.5:7b, 96.4%).

## Method

1. Loaded all 26 corpus rows.
2. **Smoke-one-before-batch** (hard rule): ran ONLY `s2-001` (Firm/Company, `clear_redundant`)
   first → qwen said SAME (agrees). Wiring sane → proceeded to batch.
3. For each row, prompted qwen (`/api/generate`, `temperature=0.0`), showing ONLY the two type
   names + descriptions, **never** the corpus label / category / ground truth (independence
   preserved):
   `Do the type names '<proposal>' (<desc>) and '<existing>' (<desc>) denote the same concept?
   Answer SAME or DIFFERENT on the first line, then a short reason.`
4. Mapping: corpus `ground_truth = redundant` ⇒ expected **SAME** concept; `novel` ⇒ expected
   **DIFFERENT**. Agreement = qwen's verdict matches the corpus label under this mapping.
5. Computed agreement; investigated every disagreement individually (below).

Script + full raw responses: `site2_audit_qwen_raw.json` (committed alongside).

## Result

**Agreement rate: 23 / 26 = 88.46%** — 1.54pt under the mechanical 90% bar. **All three
disagreements are on the redundant side; every novel row was confirmed distinct.**

| Category | Rows | Agreements |
|---|---|---|
| `clear_novel` | 6 | **6 / 6** |
| `band_edge_distinct` | 8 | **8 / 8** |
| `clear_redundant` | 6 | 5 / 6 |
| `subtle_redundant` | 6 | 4 / 6 |
| **Total** | **26** | **23 / 26** |

The **accept-side is 100% corroborated**: qwen independently confirmed all **14/14 novel** rows as
genuinely distinct concepts, including all 8 `band_edge_distinct` near-miss pairs. This is the
load-bearing direction for the fix's `14/14 accept` + `8/8 band-edge` gate — the cross-family model
unanimously agrees those types are distinct and must be accepted. Zero false "SAME" on any novel
row (no false-merge-risk label is questioned).

## Disagreements — full detail + judgment

All 3 are redundant-labeled rows qwen called DIFFERENT:

| id | proposal | existing | descriptions | qwen reason (summary) |
|---|---|---|---|---|
| s2-006 | Purchase | Transaction | **byte-identical** | Purchase is a narrower/specific kind of the broader Transaction |
| s2-024 | Judge | Magistrate | **byte-identical** | Magistrate is often a lower-tier judge; hierarchy differs by legal system |
| s2-025 | Employer | Company | **differ** | Employer is the hiring role; Company is the broader business entity |

### s2-006 (Purchase / Transaction) — audit-prompt artifact, label stands

The two descriptions are **byte-identical** ("A transaction in which money is exchanged for goods
or services."). qwen reasoned on the **names** (Purchase ⊂ Transaction, subtype) and did not treat
the identical definitions as decisive. The corpus labels on the **identical description**: two type
proposals whose definitions are word-for-word the same are redundant *for a type registry's dedup
purpose*. This is the exact **subtype/supertype residual risk named in ADR-065 Finding 2** (and
risk-register R2). The production model (gemma4:e4b) returns `is_same=true` on this pair (it is one
of the 11/12 the fix correctly rejects in the spike), so the corpus label matches how the real
pipeline behaves. **Label stands** (not relabeled — anti-massaging); the disagreement documents the
borderline nature, not a corpus error.

### s2-024 (Judge / Magistrate) — audit-prompt artifact, label stands (ADR-anticipated)

Descriptions again **byte-identical**. qwen distinguished on real-world judicial hierarchy
(magistrate = lower-tier). ADR-065 Finding 2 **explicitly names "Judge/Magistrate whose ground
truth is jurisdiction-dependent"** as a named residual risk. Given identical definitions, the
registry treats them as redundant; given name semantics + a specific jurisdiction, they can be
distinct. gemma4:e4b returns `is_same=true` (rejected by the fix, one of the 11/12). Genuinely
borderline, jurisdiction-dependent — **label stands**, disagreement is the anticipated residual
risk, tracked (R2 / R8).

### s2-025 (Employer / Company) — corroborates the known-weak label; already carved out

The only disagreement where the descriptions **differ**. qwen's DIFFERENT **corroborates** the
production model: gemma4:e4b *also* returns `is_same=false` here — which is precisely why s2-025 is
the single row carved out **by id** as the permitted redundant miss in the Phase 2 gate, and why a
Phase 5 TD tracks its adjudication-prompt tuning. Two independent model families (Qwen + Gemma)
reading Employer vs Company as distinct concepts is strong evidence that s2-025 is the **weakest
label in the corpus**. It is not fixed by the gate change (orthogonal LLM-judgment miss) and is
already isolated + tracked. **Documented, not silently accepted; not relabeled.**

## s2-026 self-referential label (ADR-065 Finding 1, re-examined)

Finding 1 flagged s2-026's label as a self-referential author shortcut. qwen independently returned
**SAME** for `Automobile Manufacturer` vs `Company` (byte-identical description copy-pasted from the
generic Company definition) — **agreeing** with the corpus `redundant` label. The cross-family model
corroborates it. **Resolved.**

## Judgment + DoD disposition

- **DoD path taken: (b) "every disagreement adjudicated + documented."** Raw agreement (88.46%) is
  1.54pt below the mechanical 90% bar; all 3 disagreements are adjudicated above with zero
  unresolved corpus-label errors.
- **No labels were changed** (anti-massaging discipline, `feedback_anti_massaging_applies`). The
  three disagreements are: 2 borderline subtype/identical-description pairs where qwen weighs
  name-hierarchy over the identical definitions the corpus + production model treat as decisive
  (the exact residual risk ADR-065 Finding 2 already names + tracks), and 1 known-weak label
  (s2-025) that cross-family confirmation *reinforces* is correctly carved out.
- **The accept-side — the fix's load-bearing direction — is 100% cross-family confirmed** (14/14
  novel, 8/8 band-edge). The gate's `14/14 accept` + `8/8 band-edge-accept` assertions rest on
  independently-verified ground truth.
- **Residual honesty:** the redundant-side ground truth for identical-description subtype pairs
  (s2-006, s2-024) is a deliberate dedup-oriented labeling choice, defensible and production-model-
  aligned but not model-family-universal. This does not corrupt the gate (production model matches
  the corpus; s2-025 is carved out), but it is the empirical face of ADR-065's named subtype-
  conflation residual risk — monitor post-enable via `DreamSummary.type_registry_merges` (R2).

**Verdict: PASS via DoD path (b).** Corpus ground truth is fit to gate the ADR-065 fix. Proceed to
Phase 1.
