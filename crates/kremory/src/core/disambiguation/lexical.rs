use std::collections::BTreeSet;

/// Minimum token-Jaccard overlap (over significant name tokens) required for two
/// entity names to be considered lexically compatible for a DESTRUCTIVE merge.
///
/// ADR-057: cosine similarity over bare entity NAMES is an unreliable identity
/// signal — anisotropic embedders (e.g. nomic-embed-text on short proper nouns)
/// return cosine 0.90–1.00 between completely unrelated names (`cos(Ria,Morocco)
/// = 1.0000`, verified `tests/spike_td080_embedder_cosine.rs`). A cosine-only
/// merge therefore collapses all entities into one canonical id and corrupts
/// every fact's subject. The fix: a destructive merge (L4 `Merge`, L5
/// canonicalization) additionally requires a DETERMINISTIC, embedder-independent
/// name-compatibility check. Cosine alone may only ever produce a NON-destructive
/// `PotentialAlias`. This makes entity resolution robust to ANY consumer embedder
/// (BYOM invariant), not just a well-calibrated one.
pub(crate) const L4_LEXICAL_JACCARD_MIN: f32 = 0.5;

/// Deterministic, embedder-independent check: are two entity names lexically
/// compatible enough to justify a DESTRUCTIVE merge (reusing one entity's id for
/// the other)? Used by BOTH destructive-merge sites — L4 `disambiguate` (this
/// module) and L5 `canonicalization` — via the same rule so the two paths cannot
/// diverge.
///
/// Compatible iff EITHER:
/// 1. **Normalized equality** — the common case (idempotent re-ingest of the same
///    name across chunks; the dominant real merge).
/// 2. **Token-Jaccard ≥ [`L4_LEXICAL_JACCARD_MIN`]** over *significant* tokens
///    (length ≥ 2 after [`normalize_name`] — drops single-initial noise like "j").
///
/// ## Measured precision/recall (Phase A corpus, ~~67~~ **68** adversarial pairs)
///
/// Count corrected 2026-08-12 (Quinn LOW-2): `kremory-eval/fixtures/entity_pairs.jsonl`
/// holds **68** rows, verified by `wc -l` and a JSON parse. The metrics below are
/// asserted live by `tests::corpus_precision_recall`, so they are current; only the
/// row count was stale, and it had been repeated forward into new docs unchecked.
///
/// See `tests::corpus_precision_recall` for the live assertion.
///
/// | Metric | Value | Threshold |
/// |---|---|---|
/// | Precision | 0.9545 (21/22) | ≥ 0.95 (hard gate) |
/// | Trivial recall | 1.00 (8/8) | ≥ 0.90 (RISK-006 guard) |
/// | Overall recall | 0.568 (21/37) | recorded only |
///
/// **Token-Jaccard capability gaps (categories where overall recall < 1.0):**
///
/// These are NOT accepted failures. They are inherent limits of a name-token
/// approach that cannot be closed by hardcoded lists — NOR by any embedding
/// technique. (DENT-001, 2026-07-02: the earlier claim here — "context-embedding
/// (ADR-058 B1) closes these" — is WRONG and has been struck. B1 was empirically
/// KILLED: embedding `name + context` as the PRIMARY identity signal for entity
/// instances scored F1 0.051 vs bare-name 0.528 — a ~10x regression — because the
/// same entity recurs across divergent contexts, ADR-058 B1 probe 2026-06-30. R3
/// (embedding-identity-degeneracy swarm) confirms NO surveyed embedding technique
/// discriminates bare proper nouns; the signal is genuinely absent from the vector
/// space.) The ratified mechanism for these gaps is ADR-063 **Site #5**
/// (`core::dream::acronym_nickname_recall`): a deterministic structural pre-filter
/// (initialism test OR graph co-occurrence) nominates candidates → margin-triggered
/// LLM adjudication in the latency-tolerant dream phase → a deterministic write-gate
/// consumes the verdict (the LLM never authorizes a destructive write alone).
///
/// - `acronym` (IBM / International Business Machines): Jaccard 0/4 = 0 — zero
///   shared tokens regardless of threshold. Closed by Site #5's initialism
///   pre-filter + LLM adjudication, NOT by embedding.
/// - `nickname` (Bob / Robert): zero shared tokens. Closed by Site #5's
///   co-occurrence pre-filter + LLM adjudication (world knowledge), NOT by embedding.
/// - `diacritic` (café / cafe): unicode-aware `is_alphanumeric` preserves
///   diacritics, so tokens never collide → gap when the diacritic is the only
///   difference. Multi-token pairs ("Café de Flore" / "Cafe de Flore") pass via
///   shared "de"+"flore". Single-token diacritic pairs remain a name-token gap
///   (candidate for a normalization-time fold, out of ADR-063's scope).
///
/// **Known FP (homonym — measured, documented, routed to LLM adjudication):**
/// - `Amazon` / `Amazon River`: Jaccard 1/2 = 0.5 → gate TRUE despite
///   `should_merge=false`. Name tokens alone cannot distinguish homonyms. Under
///   ADR-063, a margin-triggered LLM adjudication with the entities' context
///   (Site #5 / Site #2 write-gate) resolves the homonym via world knowledge —
///   the LLM sees "Amazon River" (geography) is not "Amazon" (e-commerce) — rather
///   than any embedding-cosine test, which R3 shows cannot separate them reliably.
///
/// Worked spot-checks against verified embedder data (`tests/spike_td080_embedder_cosine.rs`):
/// - `Ria`/`Morocco`, `Ria`/`Amazon Robotics`, `Northeastern University`/`Amazon
///   Robotics` → zero shared tokens → Jaccard 0 → **incompatible** (catastrophe blocked).
/// - `Alice Johnson`/`Alice Marie Johnson` → Jaccard 2/3 = 0.67 → **compatible**.
/// - `Boston`/`Boston Consulting Group` → Jaccard 1/3 = 0.33 → **incompatible**.
pub(crate) fn names_lexically_compatible(a: &str, b: &str) -> bool {
    let na = crate::core::resolver::normalize_name(a);
    let nb = crate::core::resolver::normalize_name(b);
    if na == nb {
        return true;
    }
    // Temporal/numeric discriminators must match EXACTLY before token-Jaccard is
    // consulted at all. See `temporal_discriminators` for why Jaccard cannot be
    // trusted to see this class, and why the check must precede the significance
    // filter rather than follow it.
    if temporal_discriminators(&na) != temporal_discriminators(&nb) {
        return false;
    }
    let ta: BTreeSet<&str> = na
        .split_whitespace()
        .filter(|t| t.chars().count() >= 2)
        .collect();
    let tb: BTreeSet<&str> = nb
        .split_whitespace()
        .filter(|t| t.chars().count() >= 2)
        .collect();
    if ta.is_empty() || tb.is_empty() {
        // No significant tokens on one side; equality (handled above) is the only
        // path to compatibility — single initials etc. are never auto-merged.
        return false;
    }
    let inter = ta.intersection(&tb).count();
    let union = ta.union(&tb).count();
    union > 0 && (inter as f32 / union as f32) >= L4_LEXICAL_JACCARD_MIN
}

/// Do two names carry CONFLICTING temporal/numeric identity tokens?
///
/// The same rule [`names_lexically_compatible`] applies, exposed on its own for
/// destructive-merge paths that must NOT run the token-Jaccard arm. Site #5
/// (`acronym_nickname_recall`, ADR-063) is exactly that case: acronym and nickname
/// pairs (`IBM` / `International Business Machines`) share **zero** tokens by
/// construction, so Jaccard would reject precisely the merges that site exists for
/// — but a timestamp collapsing onto its bare time (`1037 am on 27 june 2023` →
/// `1037 am`) must still be blocked. TD-212.
///
/// Wired into the shared `identity_verdict::write_gate` as its row-0 veto, so it
/// applies at Site #5 AND Site #3. **Both were evidence-checked before wiring, not
/// enabled by analogy:**
///
/// | site | corpus | should-merge rows | blocked by the veto |
/// |---|---|---|---|
/// | #5 acronym/nickname | `site5_acronym_adversarial.jsonl` (140) | 101 non-`false` | **0** |
/// | #3 type collapse | `site3_type_collapse_adversarial.jsonl` (119) | 57 | **0** |
///
/// Site #3 was initially left OUT on the theory that a trailing numeral is a
/// legitimate type variant (`person_2` vs `person`). The corpus refuted it, and all
/// 152 real `entity_types` names carry zero numerals or month words — so the veto is
/// inert there today and earns its place as defence-in-depth for future type names
/// that DO carry a date (`meeting_2023`). Both corpora keep their temporal tokens in
/// description/context fields, never in the compared names.
pub(crate) fn temporal_conflict(a: &str, b: &str) -> bool {
    let na = crate::core::resolver::normalize_name(a);
    let nb = crate::core::resolver::normalize_name(b);
    temporal_discriminators(&na) != temporal_discriminators(&nb)
}

/// Month words. Identity-bearing in exactly the way numerals are: `23 october 2023`
/// and `23 march 2023` share day AND year and differ only here, so a numerals-only
/// rule lets that one pair through (measured: 1 of 79 — see below).
const MONTH_WORDS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// Tokens that ESTABLISH identity rather than describe it: every maximal digit run,
/// plus any month word. Two names carrying different sets of these are different
/// things, however much surrounding text they share.
///
/// ## Why this cannot be left to token-Jaccard (ADR-057's deterministic check)
///
/// Jaccard is a *proportion of shared tokens*, so a one-token difference is diluted
/// by everything the names have in common — and dates are mostly common tokens.
/// Worse, the `chars().count() >= 2` significance filter in
/// [`names_lexically_compatible`] (which exists to drop single-initial noise like
/// "j") **deletes a single-digit day**, so it both removes the discriminating token
/// AND shrinks the union, *inflating* the score:
///
/// | pair | tokens compared | Jaccard | gate |
/// |---|---|---|---|
/// | `12 july 2023` / `3 july 2023` | `{12,july,2023}` vs `{july,2023}` — `3` dropped | 2/3 = **0.667** | passed ✗ |
/// | `28 august 2023` / `23 august 2023` | `{28,august,2023}` vs `{23,august,2023}` | 2/4 = **0.500** | passed ✗ (exactly at threshold) |
///
/// This is why the check runs BEFORE the significance filter, not after it.
///
/// ## Measured (A1.9.4, `graph_mutation_log` on `.context/full-corpus.db`)
///
/// 113 merges; **93 (82.3%) destroyed temporal information**. At the `canonicalize`
/// site there were 79 distinct corrupting pairs, of which **77 passed this gate** —
/// reproduced as the RED failure of `real_corpus_date_merges_are_lexically_blocked`
/// (exit 100, "77 of 79 real corrupting merges still pass"), independently matching
/// a separate simulation over the same rows.
///
/// ## Scope, stated so it is not over-credited
///
/// ⚠️ **Corrected after adversarial review of `af7eb720`.** That commit, and this
/// comment, originally claimed the rule "closes the `canonicalize`/L5 path only".
/// **That was FALSE** — inherited from a planning doc and repeated without running
/// the grep. [`names_lexically_compatible`] has **four** production call sites, and
/// the rule is live and identical at all of them:
///
/// | path | call site |
/// |---|---|
/// | L4 merge | `disambiguation/mod.rs:488` |
/// | L4 potential-alias | `disambiguation/mod.rs:503` |
/// | L7 alias-confirmation | `disambiguation/mod.rs:764` |
/// | L5 canonicalize | `canonicalization.rs:185` |
///
/// That is the intended design (see this module's header: one rule, every
/// destructive path, "so the two paths cannot diverge"), and it makes the fix
/// BROADER than claimed — but the claim was still wrong, and a wrong architectural
/// fact in a register is what stops the next reader checking.
///
/// What is genuinely NOT covered: `site5_acronym_nickname` (10 of 21 merges collapse
/// a timestamp onto its bare time) reaches `apply_merge` via ADR-063's structural
/// pre-filter and the shared `identity_verdict::write_gate`, and never calls this
/// function — TD-212. Nor does this address the documented `Amazon`/`Amazon River`
/// homonym FP, which carries no temporal tokens and is routed to LLM adjudication
/// by design.
///
/// False-positive safety was verified in BOTH directions before shipping: 0 of 79
/// corrupting pairs survive, and 0 of the module's documented-compatible pairs are
/// newly blocked (`Alice Johnson`/`Alice Marie Johnson`, case-only differences,
/// `iPhone 12 Pro`, `May Department Stores` — the last confirming a month word used
/// as an ordinary name is unaffected, because both sides carry it equally).
fn temporal_discriminators(s: &str) -> BTreeSet<String> {
    /// Leading zeros are FORMATTING, not identity: `3 july 2023` and
    /// `03 july 2023` are the same day. Without this, the rule blocks a merge
    /// that previously succeeded — an over-blocking regression, which this
    /// codebase treats as a correctness failure in a safety control, not a
    /// tolerable conservatism (Quinn review of `af7eb720`, finding MED-2).
    fn canonical_number(digits: &str) -> String {
        let trimmed = digits.trim_start_matches('0');
        if trimmed.is_empty() {
            "0".to_string()
        } else {
            trimmed.to_string()
        }
    }

    let mut out: BTreeSet<String> = BTreeSet::new();
    let mut digits = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else if !digits.is_empty() {
            out.insert(canonical_number(&std::mem::take(&mut digits)));
        }
    }
    if !digits.is_empty() {
        out.insert(canonical_number(&digits));
    }
    // `eq_ignore_ascii_case` rather than `to_lowercase()`: the input is already
    // normalized, so the allocation was dead on every token (Quinn LOW-1). The
    // canonical lowercase form is inserted so the set is spelling-independent.
    for token in s.split_whitespace() {
        if let Some(month) = MONTH_WORDS.iter().find(|m| token.eq_ignore_ascii_case(m)) {
            out.insert((*month).to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The REAL corrupting merges, extracted from `graph_mutation_log` on
    /// `.context/full-corpus.db` (113 merges; 92 at the `canonicalize` site; 79
    /// distinct date-shaped loser→keeper pairs below). A1.4/A1.9.4 measured 82.3%
    /// of all merges as temporally corrupting.
    ///
    /// This is deliberately the FULL measured set rather than a hand-picked list:
    /// a seven-pair test passes while ~72 identical failures survive.
    const REAL_CORRUPTING_MERGES: &[(&str, &str)] = &[
    ("12 july 2023", "3 july 2023"),
    ("20 july 2023", "3 july 2023"),
    ("17 july 2023", "3 july 2023"),
    ("28 august 2023", "23 august 2023"),
    ("6 july 2023", "3 july 2023"),
    ("15 july 2023", "3 july 2023"),
    ("19 june 2023", "13 june 2023"),
    ("9 april 2023", "3 april 2023"),
    ("4 february 2023", "1 february 2023"),
    ("21 june 2023", "13 june 2023"),
    ("5 july 2023", "3 july 2023"),
    ("27 june 2023", "3 june 2023"),
    ("22 july 2023", "3 july 2023"),
    ("11 august 2023", "3 august 2023"),
    ("7 april 2023", "2 april 2023"),
    ("6 may 2023", "4 may 2023"),
    ("10 april 2023", "2 april 2023"),
    ("31 july 2023", "3 july 2023"),
    ("9 january 2023", "1 january 2023"),
    ("13 august 2023", "3 august 2023"),
    ("22 december 2022", "17 december 2022"),
    ("25 february 2023", "5 february 2023"),
    ("12 june 2023", "3 june 2023"),
    ("5 august 2023", "3 august 2023"),
    ("9 august 2023", "3 august 2023"),
    ("7 july 2023", "3 july 2023"),
    ("20 may 2022", "2 may 2022"),
    ("22 august 2022", "14 august 2022"),
    ("9 november 2022", "4 november 2022"),
    ("23 january 2022", "21 january 2022"),
    ("11 november 2022", "4 november 2022"),
    ("9 october 2022", "6 october 2022"),
    ("17 april 2022", "15 april 2022"),
    ("7 november 2022", "4 november 2022"),
    ("26 august 2023", "2 august 2023"),
    ("6 december 2023", "1 december 2023"),
    ("11 november 2023", "2 august 2023"),
    ("8 december 2023", "1 december 2023"),
    ("31 august 2023", "2 august 2023"),
    ("11 august 2023", "2 august 2023"),
    ("17 august 2023", "2 august 2023"),
    ("7 january 2024", "2 january 2024"),
    ("11 may 2023", "3 may 2023"),
    ("6 may 2023", "3 may 2023"),
    ("24 september 2023", "6 september 2023"),
    ("6 october 2023", "1 october 2023"),
    ("19 june 2022", "13 june 2022"),
    ("22 july 2022", "9 july 2022"),
    ("21 august 2022", "6 august 2022"),
    ("23 april 2022", "12 april 2022"),
    ("4 september 2022", "1 september 2022"),
    ("20 march 2022", "17 march 2022"),
    ("11 may 2022", "4 may 2022"),
    ("18 september 2022", "1 september 2022"),
    ("29 april 2022", "12 april 2022"),
    ("10 august 2022", "6 august 2022"),
    ("12 august 2023", "1 august 2023"),
    ("8 september 2023", "3 september 2023"),
    ("24 august 2023", "1 august 2023"),
    ("30 august 2023", "16 august 2023"),
    ("15 september 2023", "3 september 2023"),
    ("17 september 2023", "3 september 2023"),
    ("6 september 2023", "3 september 2023"),
    ("25 february 2023", "1 february 2023"),
    ("26 august 2023", "16 august 2023"),
    ("19 august 2023", "16 august 2023"),
    ("8 october 2023", "6 october 2023"),
    ("10 january 2024", "6 january 2024"),
    ("11 january 2024", "6 january 2024"),
    ("19 august 2023", "13 august 2023"),
    ("31 december 2023", "26 december 2023"),
    ("14 august 2023", "3 august 2023"),
    ("3 may 2023", "1 may 2023"),
    ("22 august 2023", "3 august 2023"),
    ("25 october 2023", "23 march 2023"),
    ("26 march 2023", "23 march 2023"),
    ("23 october 2023", "23 march 2023"),
    ("31 may 2023", "1 may 2023"),
    ("29 october 2023", "19 october 2023"),
    ];

    /// RED before the numeric-token rule: `12 july 2023` vs `3 july 2023` tokenises
    /// to {12, july, 2023} vs {july, 2023} — the single-char `3` is dropped by the
    /// `count() >= 2` filter, which also SHRINKS the union — giving 2/3 = 0.667,
    /// over the 0.5 gate. `28 august 2023` vs `23 august 2023` lands on exactly 0.5.
    #[test]
    fn real_corpus_date_merges_are_lexically_blocked() {
        let leaked: Vec<_> = REAL_CORRUPTING_MERGES
            .iter()
            .filter(|(loser, keeper)| names_lexically_compatible(loser, keeper))
            .collect();
        assert!(
            leaked.is_empty(),
            "{} of {} real corrupting merges still pass the lexical gate; first 5: {:?}",
            leaked.len(),
            REAL_CORRUPTING_MERGES.len(),
            &leaked[..leaked.len().min(5)]
        );
    }

    /// Guard the other direction: the fix must not block merges the module
    /// documents as CORRECT. Without this, "block everything" passes the test above.
    #[test]
    fn documented_compatible_pairs_still_merge() {
        // lexical.rs:79 — worked spot-check against verified embedder data.
        assert!(names_lexically_compatible("Alice Johnson", "Alice Marie Johnson"));
        // Normalized equality — the dominant real merge (idempotent re-ingest).
        assert!(names_lexically_compatible("3 July 2023", "3 july 2023"));
        // Identical numerals must not be treated as a conflict.
        assert!(names_lexically_compatible("iPhone 12 Pro", "iphone 12 pro"));
    }

    /// Regression guard for the over-blocking defect found by adversarial review of
    /// `af7eb720` (Quinn MED-2). The first cut of the temporal rule treated `3` and
    /// `03` as different identity tokens, so two spellings of the SAME date stopped
    /// merging — a merge that succeeded before the fix. It was in neither validation
    /// corpus, which is why it survived RED/GREEN and the full gate.
    ///
    /// Failure direction was safe (a missed consolidation, not corrupted data), but
    /// an over-block in a safety control is a correctness failure here, not
    /// acceptable conservatism: leading zeros are formatting, not identity.
    #[test]
    fn zero_padded_dates_are_the_same_date() {
        assert!(names_lexically_compatible("3 july 2023", "03 july 2023"));
        assert!(names_lexically_compatible("3 july 2023", "003 july 2023"));
        assert!(names_lexically_compatible("1037 am on 09 october 2022", "1037 am on 9 october 2022"));
        // ...and the fix must not have made DIFFERENT days equal again.
        assert!(!names_lexically_compatible("03 july 2023", "30 july 2023"));
        assert!(!names_lexically_compatible("12 july 2023", "3 july 2023"));
    }

    /// Pin the documented behaviour this fix must NOT change, so the change stays scoped.
    #[test]
    fn documented_incompatible_pairs_remain_incompatible() {
        // lexical.rs:77-80 — catastrophe cases the gate exists to block.
        assert!(!names_lexically_compatible("Ria", "Morocco"));
        assert!(!names_lexically_compatible("Boston", "Boston Consulting Group"));
    }

    /// Known FP, measured and documented at lexical.rs:69. It is routed to LLM
    /// adjudication under ADR-063, NOT to this gate. Pinned so the numeric rule is
    /// not silently credited with fixing it.
    #[test]
    fn known_homonym_false_positive_is_unchanged() {
        assert!(names_lexically_compatible("Amazon", "Amazon River"));
    }
}
