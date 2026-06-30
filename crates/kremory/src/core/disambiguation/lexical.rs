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
/// ## Measured precision/recall (Phase A corpus, 67 adversarial pairs)
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
/// approach that cannot be closed by hardcoded lists. The correct mechanism is
/// context-embedding (ADR-058 B1): embed `name + "\n" + context`, where semantic
/// distance closes gaps that pure token overlap cannot.
///
/// - `acronym` (IBM / International Business Machines): Jaccard 0/4 = 0 — zero
///   shared tokens regardless of threshold. B1 closes this via semantic embedding.
/// - `nickname` (Bob / Robert): zero shared tokens. B1 closes via embedding.
/// - `diacritic` (café / cafe): unicode-aware `is_alphanumeric` preserves
///   diacritics, so tokens never collide → gap when the diacritic is the only
///   difference. Multi-token pairs ("Café de Flore" / "Cafe de Flore") pass via
///   shared "de"+"flore". B1 closes single-token diacritic pairs.
///
/// **Known FP (homonym — measured, documented, routed to B1):**
/// - `Amazon` / `Amazon River`: Jaccard 1/2 = 0.5 → gate TRUE despite
///   `should_merge=false`. Name tokens alone cannot distinguish homonyms. B1
///   embeds `name + context`, where "Amazon River" context (geography, South
///   America, flow) diverges from "Amazon" (e-commerce) — cosine drops below
///   the merge threshold, resolving the homonym without a curated block-list.
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
