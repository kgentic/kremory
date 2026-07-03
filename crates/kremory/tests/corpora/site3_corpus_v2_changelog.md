# Site #3 (type-registry collapse) adversarial corpus — v1 → v2 changelog

**File**: `crates/kremory/tests/corpora/site3_type_collapse_adversarial.jsonl`
**Governing rule (anti-massaging)**: labels encode verifiable real-world conceptual-identity fact. A label is corrected ONLY when it is factually wrong — never because a model would merge/reject the pair. Every change below is justified on factual grounds (dictionary sense division, hypernym/hyponym relationship, or a degenerate non-word substitution), not on "what an embedder would do."

Row count is unchanged: **119 rows before, 119 rows after** (no additions, no deletions — 8 rows corrected in place, 2 rows had their `a`/`b`/descriptions replaced because the original pair was degenerate rather than a genuine test case).

## Audit outcome summary

| Disposition | Count |
|---|---|
| Rows verified correct, unchanged | 109 |
| Rows relabeled (same_concept flipped + category moved) | 8 |
| Rows replaced (degenerate non-word pair → real distinct-concept pair) | 2 |
| **Total rows** | **119** |

Category counts after v2:

| Category | v1 count | v2 count |
|---|---|---|
| trivial_duplicate | 25 | 25 (unchanged) |
| semantic_near_dup_zero_lexical | 25 | 19 (−6, moved to band_edge_moderate) |
| distinct_lemma_collision | 22 | 22 (unchanged — 2 rows replaced in place, count stable) |
| band_edge_moderate | 18 | 26 (+8, absorbed all corrections) |
| distinct_unrelated | 15 | 15 (unchanged) |
| hard_true_collapse | 8 | 6 (−2, moved to band_edge_moderate) |
| unicode_casing_variant | 6 | 6 (unchanged) |

`same_concept=true`: 64 → 56. `same_concept=false`: 55 → 63.

---

## 1. Rows the task brief flagged as "known mislabels" — AUDITED AND REJECTED (kept as-is)

The task brief asserted two rows were mislabeled. On factual verification (dictionary sense-division check via research sub-agent, cross-checked against the row's own `desc_a`/`desc_b` content), **both assertions were themselves incorrect**. Per the anti-massaging governing rule, a label is corrected only when it is factually wrong — these rows are factually correct as written, so they are **NOT changed**.

### s3-049 — Colours / Colour — KEPT AS distinct (same_concept=false)

- The brief's premise: "British plural vs singular of 'colour'" — i.e., treats the row as if `b` differs from `a` only by pluralization of the general hue property.
- The actual row content: `desc_a` = "A flag or emblem representing a military unit, ship, or nation, carried or displayed with honor" (the specialized military/naval-ensign sense of "the colours" — MW sense division: "colours" as a fixed, lexicalized plural-only noun for a flag, e.g. "sound the colours", "regimental colours"). `desc_b` = the general visual-property sense of "colour."
- Verification: this specialized "colours" (flag) sense is a real, dictionary-attested, distinct concept from the general "colour" (hue) sense — historically derived (a flag is identified by its colour scheme) but synchronically lexicalized into a separate, non-compositional sense, exactly analogous to how "Species"/"Specie" (s3-030, the category's canonical example) diverge via a naive lemma strip landing on a genuine alternate sense.
- **Verdict: row is factually correct. Not changed.**

### s3-088 — Suspenders / Suspender — KEPT AS distinct (same_concept=false)

- The brief's premise: "SAME concept (plural vs singular of the clothing item)."
- The actual row content: `desc_a` = US sense (over-the-shoulder trouser straps). `desc_b` = "In British English, a strap attached to a garter belt to hold up stockings; a distinct single garment accessory."
- Verification: this is a genuine transatlantic homonym clash, not singular/plural of one referent. British English calls the American "suspenders" **"braces"**; American English calls the British "suspender(s)" **"garters"**. Both regional senses are natively used in the plural (a garter belt has multiple suspender straps in British usage too) — so US "suspenders" and UK "suspenders" are homographs for two different garments, not a singular/plural pair of the same garment.
- **Verdict: row is factually correct. Not changed.**

These two rows demonstrate the corpus category is working as intended: `distinct_lemma_collision` exists precisely to catch cases where a naive plural/singular strip crosses into an unrelated real sense, and both rows do exactly that.

---

## 2. Degenerate non-word pairs — REPLACED

The task brief correctly identified two rows testing a **naive-strip-produces-a-non-word** failure mode rather than a genuine two-real-words distinct-concept collision. Replaced with pairs where both `a` and `b` are real, dictionary-attested English words with genuinely different meanings.

### s3-037 — was Series/Serie → now **Stocks / Stock**

- Problem with original: "Serie" is not a standard English word (occasional mistaken back-formation). "Series" is also already invariant singular/plural in English. The pair tested "strip trailing s → garbage token", not "strip trailing s → real word, wrong concept."
- Replacement: **Stocks** ("A historical wooden restraining device with holes for the ankles [or wrists], used to publicly punish and humiliate offenders") vs **Stock** ("An ownership share in a company, or a supply of goods held for sale; a financial or commercial asset"). Both are real, dictionary-attested words. Verified distinct: the historical punishment device and the financial/inventory asset share no conceptual overlap beyond the naive lemma match.

### s3-038 — was Lens/Len → now **Bellows / Bellow**

- Problem with original: "Len" is a personal name/nickname (short for Leonard/Leonora), not a common noun. The pair tested "strip trailing s → a name", not "strip trailing s → a real common noun with a different meaning."
- Replacement: **Bellows** ("A device with an expandable air chamber used to direct a stream of air onto a fire or into an instrument, such as a forge bellows or accordion") vs **Bellow** ("A loud, deep roaring shout or cry, typically made by a person or large animal in anger, pain, or command"). Both are real, dictionary-attested common words. Verified distinct: the air-pumping device and the vocal act/sound share no conceptual overlap beyond the naive lemma match.

Both replacements preserve `category: distinct_lemma_collision`, `same_concept: false`, and the row's `id`. `distinct_lemma_collision` count is unchanged (22 → 22).

---

## 3. Other-category audit — genuine mislabels found and corrected

Per the task's instruction to audit `semantic_near_dup_zero_lexical` (same_concept=true) and `hard_true_collapse` (same_concept=true) for pairs that are actually hypernym/hyponym (a specific TYPE of the other) or otherwise not truly identical — rather than genuine synonyms/register variants. Each of the 8 corrections below was independently verified via a Sonnet research sub-agent classifying the pair as SAME_CONCEPT / HYPERNYM_HYPONYM / RELATED_NOT_IDENTICAL, and the sub-agent confirmed HYPERNYM_HYPONYM (or a clean category-mismatch) for every one. All 8 are moved to `band_edge_moderate` (same_concept=false) — genuinely distinct concepts in a related/adjacent domain, which is exactly what that category is for.

### From `semantic_near_dup_zero_lexical` → `band_edge_moderate` (6 rows)

- **s3-022 Corporation / Enterprise** — "Corporation" names a specific legal-entity type (formed via incorporation under law). "Enterprise" is the broader business-venture category that also includes partnerships and sole proprietorships. Every corporation is an enterprise; not every enterprise is a corporation. Hypernym/hyponym, not synonymy.
- **s3-025 Manuscript / Document** — Explicitly flagged in the task brief as an example to check. "Manuscript" is a hyponym of "Document" (a specific pre-publication literary-work instance). A manuscript is always a document; most documents (contracts, invoices, reports) are not manuscripts.
- **s3-028 Cash / Currency** — "Cash" names the physical-instantiation subtype (coins/banknotes on hand). "Currency" is the broader monetary-system concept that also covers bank-ledger balances and digital forms. Cash is a form currency can take, not a synonym for it.
- **s3-094 Cuisine / Food** — "Cuisine" names a style/tradition of food preparation (e.g. "Italian cuisine"), not the edible substance itself. "Food" is the substance. Category-mismatch (style-of-preparation vs the-thing-prepared), not conceptual identity.
- **s3-099 Statute / Law** — "Statute" is a hyponym of "Law" (specifically legislature-enacted law). "Law" is the broader category that also covers common law, case law, and regulations. A statute is always a law; not all law is statute (e.g. judge-made common law).
- **s3-100 Surgeon / Doctor** — The original row's own rationale already said "a specialization of the broader medical-doctor concept" — an explicit admission of hyponymy that contradicted its own `same_concept: true` label. "Surgeon" is a hyponym of "Doctor." A surgeon is always a doctor; most doctors are not surgeons.

### From `hard_true_collapse` → `band_edge_moderate` (2 rows)

- **s3-075 Non-profit Organization / Charity** — "Non-profit Organization" is the broad legal/tax category (includes trade associations, universities, arts bodies, research institutes). "Charity" is a specific type of non-profit focused on aid/donations. Every charity is a non-profit; not every non-profit is a charity (e.g. a professional trade association is a non-profit but not a charity).
- **s3-077 Smartphone Maker / Mobile Device Manufacturer** — "Smartphone Maker" is restricted to phones. "Mobile Device Manufacturer" is the broader category also covering tablets, wearables, and e-readers. A tablet-only manufacturer is a mobile device manufacturer but not a smartphone maker.

For each corrected row: `same_concept` flipped `true → false`, `category` moved to `band_edge_moderate`, `rationale` rewritten to state the hypernym/hyponym (or category-mismatch) relationship explicitly with a "CORRECTED (was mislabeled same_concept=true...)" prefix, and `desc_a`/`desc_b` were sharpened to make the breadth difference explicit (e.g. Currency's description now explicitly notes it "includ[es] its non-physical/digital forms" to contrast with Cash's physical-only scope).

---

## 4. Categories audited, no changes needed

- **hard_true_collapse (6 remaining rows: s3-073, s3-074, s3-076, s3-078, s3-079, s3-083)** — verified each is a genuine same-concept pair (register/phrasing variant only, no breadth difference): Startup/Early-stage Company, Automobile Manufacturer/Car Company, Chief Executive/CEO, Law Firm/Legal Practice, Merger/Corporate Consolidation, Software Vendor/Software Company. All confirmed synonymous, not hypernym/hyponym.
- **band_edge_moderate (original 18 rows, s3-065 to s3-072, s3-084, s3-104 to s3-112)** — verified each is a genuinely distinct pair in a related/adjacent domain (e.g. Physician/Attorney, Musician/Athlete, Aircraft/Watercraft, Bacteria/Virus, Violin/Cello, Senator/Governor). All confirmed correctly labeled same_concept=false with real conceptual distinctness, not merely "related enough that a model might confuse them."
- **distinct_unrelated (15 rows, s3-050 to s3-064)** — verified each pair is from completely unrelated domains (e.g. Vehicle/Recipe, Software/Mountain, Choir/Spreadsheet). All confirmed correctly labeled.
- **trivial_duplicate (25 rows)** and **unicode_casing_variant (6 rows)** — verified each pair differs only by casing, whitespace, plural/singular, hyphenation, or diacritic, with the underlying concept genuinely identical in every case (e.g. Wristwatch/Wrist Watch, Café/Cafe, COMPANY/Company). No hypernym/hyponym or distinct-concept issues found.
- **Remaining `semantic_near_dup_zero_lexical` rows (19, after the 6 moved out)** — re-verified as genuine synonym/register pairs, not hypernym/hyponym: Human/Individual, Physician/Doctor, Firm/Company, Automobile/Car, Attorney/Lawyer, Residence/Home, Purchase/Acquisition, Kid/Child, Instructor/Teacher, Nation/Country, Cellphone/Mobile Phone, Vocation/Occupation, Illness/Disease, Educator/Teacher, Vendor/Supplier, Bicycle/Bike, Textbook/Coursebook, Warehouse/Storage Facility, Counsel/Lawyer. All confirmed same-concept, no breadth/type mismatch.

## 5. Descriptions review

Reviewed `desc_a`/`desc_b` across all 119 rows for realism and concept-fidelity (the real cosine-similarity gate reads these fields). Sharpened descriptions on the 8 corrected rows plus the 2 replaced rows to make the factual distinction explicit and machine-readable (e.g. adding "of any kind (contract, report, letter, etc.)" to Document's description so the breadth contrast with Manuscript is legible in the text itself, not just implied). All other descriptions were already accurate reflections of their concepts — same-concept pairs have genuinely similar descriptions, distinct-concept pairs have genuinely different descriptions, with no drift found elsewhere in the corpus.

## 6. Uncertain calls — resolved conservatively

- **s3-096 Illness/Disease**: there is a real clinical nuance (illness = subjective lived experience of being unwell; disease = objective pathological process with a specific cause). Considered flagging as hypernym/hyponym-adjacent, but concluded this is a genuine near-synonym pair in general (non-clinical) usage and the existing `semantic_near_dup_zero_lexical` category — kept as-is (same_concept=true), not moved. This is a softer call than the 8 corrected pairs; if a stricter clinical-register interpretation is wanted later, this row is the best candidate for re-review.
- **s3-018 Firm/Company** and **s3-022 Corporation/Enterprise**: both involve business-entity vocabulary, but Firm/Company are genuinely interchangeable in ordinary usage (no breadth asymmetry — a "firm" is not a narrower TYPE of company, it's a register variant, often used for professional-services businesses specifically but not exclusively), whereas Corporation/Enterprise have a clear breadth asymmetry (corporation implies a specific legal structure; enterprise does not). Kept Firm/Company as same_concept=true; corrected Corporation/Enterprise as documented above.
- **s3-075 Non-profit Organization/Charity** and **s3-078 Law Firm/Legal Practice**: both are phrasing pairs in the `hard_true_collapse` category. Law Firm/Legal Practice have no breadth asymmetry (a legal practice IS a law firm in every ordinary sense — same referent, different register). Non-profit/Charity has a clear breadth asymmetry (documented above) — corrected only the latter.
