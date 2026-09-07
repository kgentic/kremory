use std::collections::HashSet;
use std::sync::OnceLock;

use metrics::counter;
use tracing;
use unicode_segmentation::UnicodeSegmentation;

use crate::core::intelligence::ExtractedEntity;
use crate::core::resolver::normalize_name;

// ─── Tier-1 deterministic OnceLock caches ────────────────────────────────────
//
// These are computed once from compile-time data and never invalidated.
// No mutation path exists — the values are structurally identical across
// every process lifetime. OnceLock ensures the allocation happens exactly
// once per process even under concurrent access.
//
// Tier definitions (three-cache separation):
//   Tier-1: OnceLock<T> — deterministic, never invalidated (this section)
//   Tier-2: RwLock<HashMap> — mutable-data-derived, invalidated on DIRTY
//   Tier-3: HashSet/Vec scoped to a single function call, not stored in self

/// Lazily initialised `HashSet` view of STOP_WORDS for O(1) membership tests.
/// Tier-1 cache: computed once from the compile-time slice, never invalidated.
static STOP_WORDS_SET: OnceLock<HashSet<&'static str>> = OnceLock::new();

/// Sentence-terminating punctuation characters used to detect sentence boundaries
/// during proper-noun scanning. Tier-1 cache: side-effect-free, deterministic.
static SENTENCE_TERMINATORS: OnceLock<HashSet<char>> = OnceLock::new();

/// Minimum token length (in characters) for an entity candidate to be considered.
/// Filters single-char and two-char noise. Tier-1 cache: constant value,
/// computed once.
///
/// Not used in production code — a prior silent `effective_min`
/// clamp in `is_oov_candidate` was reverted. Retained for the OnceLock invariant test.
#[cfg(test)]
static MIN_ENTITY_TOKEN_LEN: OnceLock<usize> = OnceLock::new();

/// Return a reference to the lazily initialised stop-word set (Tier-1 cache).
fn stop_words_set() -> &'static HashSet<&'static str> {
    STOP_WORDS_SET.get_or_init(|| STOP_WORDS.iter().copied().collect())
}

/// Return a reference to the lazily initialised sentence-terminator set (Tier-1).
fn sentence_terminators() -> &'static HashSet<char> {
    SENTENCE_TERMINATORS.get_or_init(|| ['.', '?', '!'].iter().copied().collect())
}

/// Return the minimum entity token length constant (Tier-1 cache).
///
/// Not used in production code — a prior silent `effective_min`
/// clamp in `is_oov_candidate` was reverted. Retained for the OnceLock invariant test.
#[cfg(test)]
fn min_entity_token_len() -> usize {
    *MIN_ENTITY_TOKEN_LEN.get_or_init(|| 4)
}

// ─── LanguageAdapter trait ───────────────────────────────────────────────────

/// Per-language dictionary + stop-words provider.
/// Abstracts away the English-specific dependency on zspell/stop-words crate
/// so the pipeline can support multiple languages via adapter pattern.
pub trait LanguageAdapter: Send + Sync {
    fn dictionary(&self) -> &zspell::Dictionary;
    fn stop_words(&self) -> &HashSet<String>;
}

/// Common English words that appear title-cased but are not proper nouns.
const STOP_WORDS: &[&str] = &[
    "The", "A", "An", "And", "Or", "But", "In", "On", "At", "To", "For", "Of", "With", "By",
    "From", "That", "This", "These", "Those", "It", "He", "She", "They", "We", "You", "I", "My",
    "Your", "His", "Her", "Their", "Our", "Both", "All", "Each", "So", "If", "As", "Do", "Did",
    "Has", "Had", "Was", "Were", "Are", "Is", "Be", "Can", "Will", "Not", "What", "When", "Where",
    "How", "Who", "Why", "Also", "Just", "Now", "Here", "There", "Then", "Well", "Very", "Most",
    "Some", "Any", "No",
];

/// Returns true if the word is in the stop-word list (exact match — stop words are
/// stored with their natural Title-Case so comparison is direct).
/// Uses the Tier-1 OnceLock cache for O(1) membership test.
fn is_stop_word(word: &str) -> bool {
    stop_words_set().contains(word)
}

/// Returns true if the word starts with an ASCII uppercase letter.
fn is_title_case(word: &str) -> bool {
    word.chars()
        .next()
        .map(|c| c.is_uppercase())
        .unwrap_or(false)
}

/// Strip leading/trailing punctuation characters from a word token so that
/// "Phoenix." → "Phoenix" and ""Corp"" → "Corp".
fn strip_punctuation(word: &str) -> &str {
    word.trim_matches(|c: char| !c.is_alphanumeric())
}

/// Scans text for capitalized multi-word sequences (1–4 words) that are not
/// already in the `existing` entity list.  Returns candidate entities with
/// `label = "Entity"` and `properties = {"source": "proper_noun_scan"}`.
///
/// This is a domain-agnostic heuristic.  It does NOT know about meetings,
/// speakers, or any specific content format.  It simply finds proper nouns.
///
/// **Rules applied (in order):**
/// 1. Split text into tokens; track whether each token is sentence-initial
///    (preceded by start-of-text or a sentence-ending punctuation: `.` `?` `!`).
/// 2. Build runs of 1–4 consecutive Title-Case tokens (after stripping
///    surrounding punctuation), stopping the run when a non-Title-Case token
///    or stop-word is encountered.
/// 3. For single-word candidates: discard if sentence-initial *or* if the word
///    is a stop-word.
/// 4. For multi-word candidates (2+ tokens): keep even when sentence-initial,
///    provided no token in the run is a stop-word and at least one token has
///    ≥ 4 alphabetic characters.
/// 5. Discard any candidate whose normalized name already appears in `existing`.
pub fn scan_proper_nouns(text: &str, existing: &[ExtractedEntity]) -> Vec<ExtractedEntity> {
    if text.is_empty() {
        return Vec::new();
    }

    // Pre-compute the set of normalized existing names for O(1) lookup.
    let existing_normalized: std::collections::HashSet<String> =
        existing.iter().map(|e| normalize_name(&e.name)).collect();

    // ── Tokenise ──────────────────────────────────────────────────────────────
    // We need to know (a) the raw token and (b) whether it is sentence-initial.
    // A sentence boundary is defined as the *end* of a token whose stripped form
    // ends with `.`, `?`, or `!`.  The token immediately following is
    // sentence-initial.

    let raw_tokens: Vec<&str> = text.split_whitespace().collect();
    if raw_tokens.is_empty() {
        return Vec::new();
    }

    // Build parallel vec of (stripped_word, is_sentence_initial).
    let mut tokens: Vec<(&str, bool)> = Vec::with_capacity(raw_tokens.len());
    let mut next_is_sentence_initial = true; // first token is always sentence-initial

    for raw in &raw_tokens {
        let stripped = strip_punctuation(raw);
        tokens.push((stripped, next_is_sentence_initial));

        // Determine if *this* token ends a sentence.
        // Uses Tier-1 OnceLock sentence_terminators() cache.
        let ends_sentence = raw
            .chars()
            .last()
            .map(|c| sentence_terminators().contains(&c))
            .unwrap_or(false);
        next_is_sentence_initial = ends_sentence;
    }

    // ── Scan for proper-noun runs ─────────────────────────────────────────────
    let mut candidates: Vec<String> = Vec::new();
    let n = tokens.len();
    let mut i = 0;

    while i < n {
        let (word, sentence_initial) = tokens[i];

        if word.is_empty() || !is_title_case(word) || is_stop_word(word) {
            i += 1;
            continue;
        }

        // Start building a run from position i.
        let mut run: Vec<&str> = vec![word];
        let mut j = i + 1;

        while j < n && run.len() < 4 {
            let (next_word, _) = tokens[j];
            if next_word.is_empty() || !is_title_case(next_word) || is_stop_word(next_word) {
                break;
            }
            run.push(next_word);
            j += 1;
        }

        // Evaluate runs from longest to shortest so that "Acme Corp" is preferred
        // over two separate single-word candidates "Acme" and "Corp".
        let mut accepted = false;
        for end in (0..run.len()).rev() {
            let slice = &run[..=end];
            let word_count = slice.len();

            // Filter: at least one word must have ≥ 4 alphabetic characters.
            let has_long_word = slice
                .iter()
                .any(|w| w.chars().filter(|c| c.is_alphabetic()).count() >= 4);
            if !has_long_word {
                continue;
            }

            // Filter: no stop-words in this slice.
            if slice.iter().any(|w| is_stop_word(w)) {
                continue;
            }

            // Single-word rule: discard if sentence-initial.
            if word_count == 1 && sentence_initial {
                continue;
            }

            let candidate_name = slice.join(" ");
            let normalized = normalize_name(&candidate_name);

            // Filter: not already known.
            if existing_normalized.contains(&normalized) {
                continue;
            }

            candidates.push(candidate_name);
            accepted = true;
            break; // took the longest valid prefix; skip sub-sequences
        }

        // Advance past the consumed run (or just past i if nothing accepted).
        if accepted {
            i = j;
        } else {
            i += 1;
        }
    }

    // Deduplicate (preserve first occurrence).
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    candidates
        .into_iter()
        .filter(|c| seen.insert(normalize_name(c)))
        .map(|name| ExtractedEntity {
            label: "Entity".to_string(),
            properties: serde_json::json!({"name": name, "source": "proper_noun_scan"}),
            name,
        })
        .collect()
}

// ─── OOV Auditor (language-agnostic entity safety net) ────────────────────────

/// Language-agnostic entity auditor that uses dictionary subtraction (OOV detection)
/// to catch domain-specific terms the LLM may have missed. Runs post-LLM as a
/// safety net, not as a pre-filter.
///
/// Uses: zspell (Hunspell dictionary), stop-words crate, unicode-segmentation.
/// No hardcoded word lists — language knowledge comes from per-language dictionaries
/// and stop-word sets.
pub struct OovAuditor {
    dict: zspell::Dictionary,
    stop_words: HashSet<String>,
}

impl LanguageAdapter for OovAuditor {
    fn dictionary(&self) -> &zspell::Dictionary {
        &self.dict
    }
    fn stop_words(&self) -> &HashSet<String> {
        &self.stop_words
    }
}

impl OovAuditor {
    /// Create an auditor from pre-loaded dictionary and stop-word set.
    /// Dictionary and stop words are per-language — the caller provides them.
    pub fn new(dict: zspell::Dictionary, stop_words: HashSet<String>) -> Self {
        Self { dict, stop_words }
    }

    /// Returns true if `word` is an OOV candidate: not in dictionary, not a stop word,
    /// not numeric, not a contraction, and at least `min_len` chars.
    ///
    /// `min_len` is honoured directly — callers are responsible for choosing
    /// the appropriate floor. The global `min_entity_token_len()` constant
    /// (Tier-1 OnceLock) is available for callers that want it,
    /// but `is_oov_candidate` does NOT silently clamp to it. Silently overriding
    /// the caller's `min_len` with `max(min_len, 4)` was a scope-creep regression,
    /// since reverted.
    fn is_oov_candidate(&self, word: &str, min_len: usize) -> bool {
        if word.len() < min_len {
            return false;
        }
        if word.chars().all(|c| c.is_numeric() || c == '.' || c == ',') {
            return false;
        }
        let lower = word.to_lowercase();
        if self.stop_words.contains(&lower) {
            return false;
        }
        if word.contains('\'') || word.contains('\u{2019}') {
            return false;
        }
        !self.dict.check_word(word) && !self.dict.check_word(&lower)
    }

    // ─── Programmatic candidate pipeline ─────────────────────────────────────

    /// Run the full programmatic entity candidate pipeline (zero LLM):
    ///   1. scan_proper_nouns (Title-Case pattern detection)
    ///   2. OOV singles (words not in dictionary)
    ///   3. OOV runs (2–4 consecutive OOV words)
    ///   4. PMI bigrams (co-occurrence above threshold 2.0)
    ///   5. Merge + deduplicate + cap
    ///
    /// Returns raw candidate strings (not typed — LLM handles typing).
    /// Validated at 95% recall across 8 domains, <5ms, zero LLM calls.
    /// Run the language-agnostic OOV candidate pipeline (zero LLM):
    ///   1. OOV singles (words not in dictionary, ≥4 chars)
    ///   2. OOV runs (2–4 consecutive OOV words — multi-word entities)
    ///   3. Deduplicate + cap
    ///
    /// Returns raw candidate strings (not typed — LLM handles typing).
    ///
    /// Design rationale (from spike findings):
    /// - Scanner omitted: English-only (Title Case heuristic), not language-agnostic
    /// - PMI omitted: falsified at single-document scale (needs 100k+ tokens)
    /// - OOV + two-call LLM = 83% recall, best language-agnostic path
    /// - Cap at 20-25: >25 candidates overloads LLM entity confirmation call
    pub fn extract_candidates(&self, text: &str, max_candidates: usize) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut seen = HashSet::new();
        let mut result = Vec::new();

        // 1. OOV singles (min 4 chars to filter ASR noise / abbreviations)
        for word in text.unicode_words() {
            if self.is_oov_candidate(word, 4) {
                let key = word.to_lowercase();
                if seen.insert(key) {
                    result.push(word.to_string());
                }
            }
        }

        // 2. OOV runs (2–4 consecutive OOV words — catches multi-word entities)
        let words: Vec<&str> = text.unicode_words().collect();
        let mut run: Vec<&str> = Vec::new();
        for word in &words {
            if self.is_oov_candidate(word, 3) {
                run.push(word);
                if run.len() >= 4 {
                    let name = run.join(" ");
                    let key = name.to_lowercase();
                    if seen.insert(key) {
                        result.push(name);
                    }
                    run.clear();
                }
            } else {
                if run.len() >= 2 {
                    let name = run.join(" ");
                    let key = name.to_lowercase();
                    if seen.insert(key) {
                        result.push(name);
                    }
                }
                run.clear();
            }
        }
        if run.len() >= 2 {
            let name = run.join(" ");
            let key = name.to_lowercase();
            if seen.insert(key) {
                result.push(name);
            }
        }

        // 3. Cap at max_candidates (preserves insertion order)
        result.truncate(max_candidates);

        let programmatic_candidates = result.len() as u64;
        counter!("rql.extraction.programmatic_candidates").increment(programmatic_candidates);
        tracing::info!(
            programmatic_candidates,
            "kremory.extraction.programmatic_candidates"
        );
        result
    }

    // ─── Post-LLM audit (legacy safety net) ──────────────────────────────────

    /// Audit chunk text against already-known entities. Returns OOV entity candidates
    /// that the LLM missed — domain terms, technical jargon, drug names, etc.
    ///
    /// When using `ProgrammaticFirstExtractor`, this is redundant (pipeline already
    /// ran pre-LLM). Kept for backwards compatibility with LLM-first extractors.
    pub fn audit(&self, text: &str, known: &[ExtractedEntity]) -> Vec<ExtractedEntity> {
        if text.is_empty() {
            return Vec::new();
        }

        let known_normalized: HashSet<String> =
            known.iter().map(|e| normalize_name(&e.name)).collect();

        let mut seen = HashSet::new();
        let mut candidates: Vec<ExtractedEntity> = Vec::new();

        // Single-word OOV candidates
        for word in text.unicode_words() {
            if !self.is_oov_candidate(word, 4) {
                continue;
            }
            let normalized = normalize_name(word);
            if known_normalized.contains(&normalized) {
                continue;
            }
            if seen.insert(normalized) {
                let word_name = word.to_string();
                candidates.push(ExtractedEntity {
                    name: word_name.clone(),
                    label: "Entity".to_string(),
                    properties: serde_json::json!({"name": word_name, "source": "oov_audit"}),
                });
            }
        }

        // Multi-word OOV runs (2-4 consecutive OOV words)
        let words: Vec<&str> = text.unicode_words().collect();
        let mut run: Vec<&str> = Vec::new();
        for word in &words {
            if self.is_oov_candidate(word, 3) {
                run.push(word);
                if run.len() >= 4 {
                    let name = run.join(" ");
                    let normalized = normalize_name(&name);
                    if !known_normalized.contains(&normalized) && seen.insert(normalized) {
                        candidates.push(ExtractedEntity {
                            properties: serde_json::json!({"name": name, "source": "oov_audit_run"}),
                            name,
                            label: "Entity".to_string(),
                        });
                    }
                    run.clear();
                }
            } else {
                if run.len() >= 2 {
                    let name = run.join(" ");
                    let normalized = normalize_name(&name);
                    if !known_normalized.contains(&normalized) && seen.insert(normalized) {
                        candidates.push(ExtractedEntity {
                            properties: serde_json::json!({"name": name, "source": "oov_audit_run"}),
                            name,
                            label: "Entity".to_string(),
                        });
                    }
                }
                run.clear();
            }
        }
        if run.len() >= 2 {
            let name = run.join(" ");
            let normalized = normalize_name(&name);
            if !known_normalized.contains(&normalized) && seen.insert(normalized) {
                candidates.push(ExtractedEntity {
                    properties: serde_json::json!({"name": name, "source": "oov_audit_run"}),
                    name,
                    label: "Entity".to_string(),
                });
            }
        }

        let oov_audit_adds = candidates.len() as u64;
        counter!("rql.extraction.oov_audit_adds").increment(oov_audit_adds);
        tracing::info!(oov_audit_adds, "kremory.extraction.oov_audit_adds");
        candidates
    }
}

// ─── PMI collocation detection ───────────────────────────────────────────────

/// Compute PMI (Pointwise Mutual Information) for word bigrams.
/// Returns bigrams above `threshold` sorted by PMI descending.
///
/// PMI = log2(P(w1,w2) / (P(w1) * P(w2)))
/// Threshold 2.0 validated in spike: high recall as multi-word boundary detector.
/// NPMI is broken on short documents (needs 100k+ tokens) — use raw PMI only.
///
/// No production caller (only the `#[cfg(test)]` tests below) — gated to
/// test builds instead of carrying `#[allow(dead_code)]`. Re-promote to a real
/// `pub(crate)` if/when the OOV-PMI entity scorer (parking-lot) wires it.
#[cfg(test)]
#[allow(unused_imports)]
use std::collections::HashMap;

#[cfg(test)]
pub(crate) fn compute_pmi_bigrams(text: &str, threshold: f64) -> Vec<(String, f64)> {
    let words: Vec<String> = text
        .unicode_words()
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_lowercase())
        .collect();
    let n = words.len();
    if n < 2 {
        return Vec::new();
    }

    let mut unigram_freq: HashMap<&str, usize> = HashMap::new();
    for w in &words {
        *unigram_freq.entry(w.as_str()).or_default() += 1;
    }

    let mut bigram_freq: HashMap<(&str, &str), usize> = HashMap::new();
    for pair in words.windows(2) {
        *bigram_freq
            .entry((pair[0].as_str(), pair[1].as_str()))
            .or_default() += 1;
    }

    let n_f = n as f64;
    let mut results = Vec::new();
    for ((w1, w2), freq) in &bigram_freq {
        let p_w1 = *unigram_freq.get(w1).unwrap_or(&1) as f64 / n_f;
        let p_w2 = *unigram_freq.get(w2).unwrap_or(&1) as f64 / n_f;
        let p_bigram = *freq as f64 / (n - 1) as f64;

        if p_w1 > 0.0 && p_w2 > 0.0 {
            let pmi = (p_bigram / (p_w1 * p_w2)).log2();
            if pmi >= threshold {
                results.push((format!("{} {}", w1, w2), pmi));
            }
        }
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_existing(name: &str) -> ExtractedEntity {
        ExtractedEntity {
            label: "Entity".to_string(),
            name: name.to_string(),
            properties: serde_json::Value::Null,
        }
    }

    fn names(entities: &[ExtractedEntity]) -> Vec<&str> {
        entities.iter().map(|e| e.name.as_str()).collect()
    }

    /// Tier-1 OnceLock caches return the same pointer on two calls
    /// (pointer equality proves single allocation, not a copy on each access).
    #[test]
    fn oncelock_tier1_caches_same_pointer_on_repeated_calls() {
        // stop_words_set: call twice — must be same *const pointer.
        let ptr_a = stop_words_set() as *const HashSet<&'static str>;
        let ptr_b = stop_words_set() as *const HashSet<&'static str>;
        assert_eq!(
            ptr_a, ptr_b,
            "STOP_WORDS_SET OnceLock must return the same allocation"
        );

        // sentence_terminators: same check.
        let ptr_c = sentence_terminators() as *const HashSet<char>;
        let ptr_d = sentence_terminators() as *const HashSet<char>;
        assert_eq!(
            ptr_c, ptr_d,
            "SENTENCE_TERMINATORS OnceLock must return the same allocation"
        );

        // min_entity_token_len: value must be deterministic.
        let val_e = min_entity_token_len();
        let val_f = min_entity_token_len();
        assert_eq!(
            val_e, val_f,
            "MIN_ENTITY_TOKEN_LEN OnceLock must return the same value on every call"
        );
        let ptr_e = MIN_ENTITY_TOKEN_LEN.get().expect("initialized") as *const usize;
        let ptr_f = MIN_ENTITY_TOKEN_LEN.get().expect("initialized") as *const usize;
        assert_eq!(
            ptr_e, ptr_f,
            "MIN_ENTITY_TOKEN_LEN OnceLock must return the same allocation"
        );
    }

    #[test]
    fn test_scan_finds_title_case_sequences() {
        // "Alice" is sentence-initial (single word) → filtered.
        // "Acme Corp" and "New York" are mid-sentence multi-word → kept.
        let result = scan_proper_nouns("Alice works at Acme Corp in New York.", &[]);
        let found = names(&result);
        assert!(
            found.contains(&"Acme Corp"),
            "expected 'Acme Corp' in {:?}",
            found
        );
        assert!(
            found.contains(&"New York"),
            "expected 'New York' in {:?}",
            found
        );
        assert!(
            !found.contains(&"Alice"),
            "'Alice' is sentence-initial single word, must not appear in {:?}",
            found
        );
    }

    #[test]
    fn test_scan_excludes_existing_entities() {
        // "Acme Corp" is already known — should not be returned.
        let existing = vec![make_existing("Acme Corp")];
        let result = scan_proper_nouns("Alice works at Acme Corp.", &existing);
        let found = names(&result);
        assert!(
            !found.contains(&"Acme Corp"),
            "'Acme Corp' is already known, must not appear in {:?}",
            found
        );
    }

    #[test]
    fn test_scan_ignores_sentence_initial() {
        // "The" is sentence-initial stop word → filtered.
        // "Phoenix" follows "called" (mid-sentence) → kept.
        // "It" is a stop word → filtered.
        let result = scan_proper_nouns("The project is called Phoenix. It launched in March.", &[]);
        let found = names(&result);
        assert!(
            found.contains(&"Phoenix"),
            "expected 'Phoenix' in {:?}",
            found
        );
        assert!(
            !found.contains(&"The"),
            "'The' is a stop word, must not appear in {:?}",
            found
        );
        assert!(
            !found.contains(&"It"),
            "'It' is a stop word, must not appear in {:?}",
            found
        );
    }

    #[test]
    fn test_scan_filters_stop_words() {
        // "She" is sentence-initial stop word → filtered.
        // "The" is a stop word → filtered even mid-sentence if it breaks a run.
        // "Quick Brown Fox" — none are stop words, all title-case, mid-sentence → kept.
        let result = scan_proper_nouns("She said The Quick Brown Fox jumps.", &[]);
        let found = names(&result);
        assert!(
            found.contains(&"Quick Brown Fox"),
            "expected 'Quick Brown Fox' in {:?}",
            found
        );
        assert!(
            !found.contains(&"She"),
            "'She' is a stop word, must not appear in {:?}",
            found
        );
        assert!(
            !found.contains(&"The"),
            "'The' is a stop word, must not appear in {:?}",
            found
        );
    }

    #[test]
    fn test_scan_empty_text() {
        let result = scan_proper_nouns("", &[]);
        assert!(result.is_empty(), "empty text must return empty vec");
    }

    // ─── PMI tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_pmi_returns_collocations() {
        // "acme corp" appears twice as a bigram — PMI should be elevated
        let text = "Alice works at Acme Corp. Bob also joined Acme Corp last year.";
        let results = compute_pmi_bigrams(text, 0.0);
        let bigram_names: Vec<&str> = results.iter().map(|(s, _)| s.as_str()).collect();
        assert!(
            bigram_names
                .iter()
                .any(|b| b.contains("acme") && b.contains("corp")),
            "expected 'acme corp' bigram in {:?}",
            bigram_names
        );
    }

    #[test]
    fn test_pmi_empty_text() {
        let results = compute_pmi_bigrams("", 2.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_pmi_single_word() {
        let results = compute_pmi_bigrams("hello", 0.0);
        assert!(results.is_empty(), "single word should produce no bigrams");
    }

    // ─── extract_candidates tests ────────────────────────────────────────────

    fn load_test_auditor() -> OovAuditor {
        let aff = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.aff"
        ))
        .expect("en_US.aff");
        let dic = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.dic"
        ))
        .expect("en_US.dic");
        let dict = zspell::builder()
            .config_str(&aff)
            .dict_str(&dic)
            .build()
            .expect("build dictionary");
        let stops: HashSet<String> = stop_words::get(stop_words::LANGUAGE::English)
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        OovAuditor::new(dict, stops)
    }

    #[test]
    fn test_extract_candidates_catches_oov_terms() {
        let auditor = load_test_auditor();
        // "Krishnamurthy" is a proper name, not in en_US.dic → OOV candidate
        let candidates = auditor.extract_candidates(
            "The meeting was led by Krishnamurthy from the engineering team.",
            25,
        );
        let lower: Vec<String> = candidates.iter().map(|c| c.to_lowercase()).collect();
        assert!(
            lower.iter().any(|c| c.contains("krishnamurthy")),
            "expected 'Krishnamurthy' as OOV candidate in {:?}",
            candidates
        );
    }

    #[test]
    fn test_extract_candidates_catches_multi_word_oov() {
        let auditor = load_test_auditor();
        // Two consecutive OOV words should form a run candidate
        let candidates = auditor.extract_candidates(
            "The doctor prescribed Atorvastatin Hydroxychloroquine combination therapy.",
            25,
        );
        // At least one of the drug names should appear as OOV
        assert!(
            !candidates.is_empty(),
            "expected OOV drug names as candidates"
        );
    }

    #[test]
    fn test_extract_candidates_skips_dictionary_words() {
        let auditor = load_test_auditor();
        // Common English words should NOT appear as candidates
        let candidates =
            auditor.extract_candidates("The quick brown fox jumps over the lazy dog.", 25);
        let lower: Vec<String> = candidates.iter().map(|c| c.to_lowercase()).collect();
        assert!(
            !lower.contains(&"quick".to_string()),
            "'quick' is a dictionary word, should not be a candidate"
        );
        assert!(
            !lower.contains(&"brown".to_string()),
            "'brown' is a dictionary word, should not be a candidate"
        );
    }

    #[test]
    fn test_extract_candidates_respects_cap() {
        let auditor = load_test_auditor();
        let candidates = auditor.extract_candidates(
            "Alice from Acme Corp met Bob at Zenith Dynamics. They discussed TechCorp initiatives with GlobalNet partners.",
            3,
        );
        assert!(
            candidates.len() <= 3,
            "expected at most 3 candidates, got {}",
            candidates.len()
        );
    }

    #[test]
    fn test_extract_candidates_empty_text() {
        let auditor = load_test_auditor();
        let candidates = auditor.extract_candidates("", 25);
        assert!(candidates.is_empty());
    }
}

// ─── UTF-8 boundary-safe slicing ─────────────────────────────────────────────
//
// Slicing a `&str` at a byte offset that is not a character boundary PANICS.
// Every offset computed by arithmetic — a word-length sum, a `- 200` search
// window, a `.min(100)` truncation, a fixed `8000` cap — can land mid-character
// on any non-ASCII input, and LLM responses and user episodes are both
// arbitrary UTF-8.
//
// This is a RECURRING defect class in this crate, not a one-off:
//   * `core/extraction_window.rs` crashed kremory-http mid-run against
//     LongMemEval — `byte index 6809 is not a char boundary; it is
//     inside 'è'` (Catalan), and again on an emoji. It took the whole server
//     down because the panic unwinds a tokio worker.
//   * `core/ingest/helpers.rs` had ALREADY solved it with its own private
//     floor/ceil helpers — the guard existed in one module and the sibling
//     module never inherited it.
//   * `core/extraction/structured.rs` and
//     `core/dream/consistency_check/audit.rs` were both still unguarded.
//
// Four private copies of the same idea is how that drift happened, so this is
// the ONE implementation. Import it; do not re-roll it locally.
//
// LLM output is untrusted UTF-8 — a panic here crashes every `remember()`
// call with no recovery path, which makes this the top fuzz-testing priority
// for this crate.

/// Round `i` DOWN to the nearest UTF-8 character boundary in `s`.
///
/// Clamps to `s.len()` first, so an out-of-range index is safe. Rounding down
/// (never up) guarantees the result is `<= i`, which callers relying on
/// `pos < s.len()` depend on.
pub(crate) fn floor_char_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Truncate `s` to at most `max_bytes`, never splitting a character.
///
/// Returns the whole string when it already fits, so callers can use it
/// unconditionally without a length check of their own — a length check is
/// exactly what tends to be written in BYTES and then get this wrong.
pub(crate) fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> &str {
    &s[..floor_char_boundary(s, max_bytes)]
}

#[cfg(test)]
mod utf8_boundary_util_tests {
    use super::*;

    #[test]
    fn floor_never_returns_a_non_boundary() {
        for s in ["aè b€ c𝄞 d", "🤔🤔🤔", "Català València", "plain ascii"] {
            for i in 0..=s.len() + 4 {
                let f = floor_char_boundary(s, i);
                assert!(s.is_char_boundary(f), "{s:?} @{i} -> {f} is not a boundary");
                assert!(f <= i.min(s.len()), "rounded UP: {f} > {i}");
                let _ = &s[..f]; // panics if f is not a boundary
            }
        }
    }

    #[test]
    fn floor_is_identity_on_ascii() {
        let s = "the quick brown fox";
        for i in 0..=s.len() {
            assert_eq!(floor_char_boundary(s, i), i);
        }
    }

    #[test]
    fn truncate_never_splits_a_character() {
        // 'è' is 2 bytes; truncating at 1 must yield "" not a split char.
        assert_eq!(truncate_on_char_boundary("è", 1), "");
        assert_eq!(truncate_on_char_boundary("è", 2), "è");
        // Emoji is 4 bytes — every interior offset must round down to 0.
        for n in 0..4 {
            assert_eq!(truncate_on_char_boundary("🤔", n), "");
        }
        assert_eq!(truncate_on_char_boundary("🤔", 4), "🤔");
        // Already-fits case returns the whole string.
        assert_eq!(truncate_on_char_boundary("abc", 999), "abc");
    }

    #[test]
    fn truncate_reproduces_the_two_live_crash_sites() {
        // structured.rs:635 shape — `&trimmed[..trimmed.len().min(100)]`
        // 'é' is 2 bytes, so its boundaries are all EVEN and byte 100 would be
        // safe by luck — the one-byte 'x' prefix shifts them odd so byte 100
        // genuinely straddles a character. (The assertion below caught this
        // when the fixture was wrong, which is the point of asserting it.)
        let llm_garbage = format!("x{}", "é".repeat(80)); // 161 bytes
        assert!(
            !llm_garbage.is_char_boundary(100),
            "fixture must straddle byte 100"
        );
        let _ = truncate_on_char_boundary(&llm_garbage, 100); // pre-fix: panic

        // audit.rs:293 shape — `&s[..8000]` guarded by a BYTE length check
        let episode = "ü".repeat(5000); // 10,000 bytes; byte 8000 is a boundary…
        let episode = format!("x{episode}"); // …shift by 1 so it is not
        assert!(
            !episode.is_char_boundary(8000),
            "fixture must straddle byte 8000"
        );
        let _ = truncate_on_char_boundary(&episode, 8000); // pre-fix: panic
    }
}
