/// Spike: OOV Dictionary Subtraction + PMI Collocation Detection
///
/// Tests whether zspell (Hunspell) + PMI can identify entity candidates
/// at comparable recall to scan_proper_nouns(), without relying on
/// English-specific Title Case heuristics.
///
/// Run with:
///   cargo test -p rql-core --test spike_oov_pmi_extraction -- --nocapture
///
/// Requires: rql-core/fixtures/dictionaries/en_US.{aff,dic}

mod spike {
    use std::collections::{HashMap, HashSet};
    use rql_core::text_utils::scan_proper_nouns;
    use unicode_segmentation::UnicodeSegmentation;

    // ── Fixture loading (same pattern as spike_programmatic_extraction) ──────

    struct Fixture {
        name: &'static str,
        key: &'static str,
        path: &'static str,
    }

    fn all_fixtures() -> Vec<Fixture> {
        vec![
            Fixture {
                name: "Mock Interview",
                key: "mock_interview",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/mock_interview.txt"),
            },
            Fixture {
                name: "Medical Consultation",
                key: "medical_consultation",
                path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/medical_consultation.txt"
                ),
            },
            Fixture {
                name: "Legal Deposition",
                key: "legal_deposition",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/legal_deposition.txt"),
            },
            Fixture {
                name: "Tech Standup",
                key: "tech_standup",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/tech_standup.txt"),
            },
            Fixture {
                name: "Sales Call",
                key: "sales_call",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/sales_call.txt"),
            },
            Fixture {
                name: "Podcast Interview",
                key: "podcast_interview",
                path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/podcast_interview.txt"
                ),
            },
            Fixture {
                name: "Board Meeting",
                key: "board_meeting",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/board_meeting.txt"),
            },
            Fixture {
                name: "News Article",
                key: "news_article",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/news_article.txt"),
            },
            Fixture {
                name: "Academic Lecture",
                key: "academic_lecture",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/academic_lecture.txt"),
            },
            Fixture {
                name: "Customer Support",
                key: "customer_support",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/customer_support.txt"),
            },
            Fixture {
                name: "Slack Thread",
                key: "slack_thread",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/slack_thread.txt"),
            },
            Fixture {
                name: "Product Review",
                key: "product_review",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/product_review.txt"),
            },
            Fixture {
                name: "Short Snippet",
                key: "short_snippet",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/short_snippet.txt"),
            },
            Fixture {
                name: "Long Report",
                key: "long_report",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/long_report.txt"),
            },
        ]
    }

    fn load_ground_truth() -> HashMap<String, Vec<String>> {
        let gt_path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/ground_truth.json");
        let gt_raw = std::fs::read_to_string(gt_path).expect("ground_truth.json");
        let gt: serde_json::Value = serde_json::from_str(&gt_raw).expect("parse ground_truth");
        let mut map = HashMap::new();
        for (key, domain) in gt.as_object().unwrap() {
            let entities: Vec<String> = domain["entities"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| e["name"].as_str().unwrap().to_lowercase())
                .collect();
            map.insert(key.clone(), entities);
        }
        map
    }

    // ── OOV extraction ──────────────────────────────────────────────────────

    fn load_dictionary() -> zspell::Dictionary {
        let aff_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.aff"
        );
        let dic_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.dic"
        );

        let aff = std::fs::read_to_string(aff_path).expect("en_US.aff not found");
        let dic = std::fs::read_to_string(dic_path).expect("en_US.dic not found");

        zspell::builder()
            .config_str(&aff)
            .dict_str(&dic)
            .build()
            .expect("failed to build zspell dictionary")
    }

    fn load_stop_words() -> HashSet<String> {
        stop_words::get(stop_words::LANGUAGE::English)
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Check if a word is OOV (not in dictionary, not a stop word, not numeric)
    fn is_oov(word: &str, dict: &zspell::Dictionary, stops: &HashSet<String>) -> bool {
        if word.len() < 2 {
            return false;
        }
        if word.chars().all(|c| c.is_numeric() || c == '.' || c == ',') {
            return false;
        }
        let lower = word.to_lowercase();
        if stops.contains(&lower) {
            return false;
        }
        // Check both original case and lowercase against dictionary
        !dict.check_word(word) && !dict.check_word(&lower)
    }

    /// Extract single-word OOV candidates (deduplicated)
    fn extract_oov_singles(
        text: &str,
        dict: &zspell::Dictionary,
        stops: &HashSet<String>,
    ) -> Vec<String> {
        let mut seen = HashSet::new();
        text.unicode_words()
            .filter(|w| is_oov(w, dict, stops))
            .filter(|w| seen.insert(w.to_lowercase()))
            .map(|w| w.to_string())
            .collect()
    }

    /// Build multi-word candidates from consecutive OOV words (runs of 2-4)
    fn extract_oov_runs(
        text: &str,
        dict: &zspell::Dictionary,
        stops: &HashSet<String>,
    ) -> Vec<String> {
        let words: Vec<&str> = text.unicode_words().collect();
        let mut runs = Vec::new();
        let mut current_run: Vec<&str> = Vec::new();

        for word in &words {
            if is_oov(word, dict, stops) {
                current_run.push(word);
                if current_run.len() >= 4 {
                    runs.push(current_run.join(" "));
                    current_run.clear();
                }
            } else {
                if current_run.len() >= 2 {
                    runs.push(current_run.join(" "));
                }
                current_run.clear();
            }
        }
        if current_run.len() >= 2 {
            runs.push(current_run.join(" "));
        }

        runs
    }

    // ── PMI collocation detection ───────────────────────────────────────────

    /// Compute PMI for bigrams; return those above threshold
    fn compute_pmi_bigrams(text: &str, threshold: f64) -> Vec<(String, f64)> {
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

    // ── Matching ────────────────────────────────────────────────────────────

    fn fuzzy_match(extracted: &str, expected: &str) -> bool {
        let e = extracted.to_lowercase();
        let x = expected.to_lowercase();
        e == x || e.contains(&x) || x.contains(&e)
    }

    fn recall(candidates: &[String], expected: &[String]) -> (usize, f64) {
        let found = expected
            .iter()
            .filter(|exp| candidates.iter().any(|ext| fuzzy_match(ext, exp)))
            .count();
        let recall = if expected.is_empty() {
            1.0
        } else {
            found as f64 / expected.len() as f64
        };
        (found, recall)
    }

    // ── Main spike test ─────────────────────────────────────────────────────

    #[test]
    fn spike_oov_pmi_recall() {
        let dict = load_dictionary();
        let gt = load_ground_truth();
        let stops = load_stop_words();

        let fixture_list = all_fixtures();
        let pmi_threshold = 2.0;

        eprintln!("\n═══════════════════════════════════════════════════════════════════════════");
        eprintln!("SPIKE: OOV Dictionary Subtraction + PMI Collocation Detection");
        eprintln!(
            "Method: zspell (en_US) + PMI bigrams (threshold {:.1}) + scanner baseline",
            pmi_threshold
        );
        eprintln!("═══════════════════════════════════════════════════════════════════════════\n");

        eprintln!(
            "{:<25} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "Domain", "OOV", "PMI", "OOV+PMI", "Scanner", "Time"
        );
        eprintln!("{}", "─".repeat(75));

        let mut totals = [0.0f64; 4]; // oov, pmi, combined, scanner
        let mut fixture_count = 0;

        for fixture in &fixture_list {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => {
                    eprintln!("{:<25} SKIP — no ground truth", fixture.name);
                    continue;
                }
            };

            let text = std::fs::read_to_string(fixture.path).expect("read fixture");
            let start = std::time::Instant::now();

            // === OOV: single words + consecutive runs ===
            let oov_singles = extract_oov_singles(&text, &dict, &stops);
            let oov_runs = extract_oov_runs(&text, &dict, &stops);
            let oov_all: Vec<String> = oov_singles
                .iter()
                .chain(oov_runs.iter())
                .map(|s| s.to_lowercase())
                .collect();

            // === PMI: high-PMI bigrams ===
            let pmi_bigrams = compute_pmi_bigrams(&text, pmi_threshold);
            let pmi_candidates: Vec<String> = pmi_bigrams.iter().map(|(s, _)| s.clone()).collect();

            // === Combined: union of OOV + PMI ===
            let combined_set: HashSet<String> = oov_all
                .iter()
                .chain(pmi_candidates.iter())
                .cloned()
                .collect();
            let combined: Vec<String> = combined_set.into_iter().collect();

            // === Scanner baseline ===
            let scanner_results = scan_proper_nouns(&text, &[]);
            let scanner_names: Vec<String> = scanner_results
                .iter()
                .map(|e| e.name.to_lowercase())
                .collect();

            let elapsed = start.elapsed();

            // === Recall ===
            let (_, r_oov) = recall(&oov_all, expected);
            let (_, r_pmi) = recall(&pmi_candidates, expected);
            let (_, r_combined) = recall(&combined, expected);
            let (_, r_scanner) = recall(&scanner_names, expected);

            eprintln!(
                "{:<25} {:>7.0}% {:>7.0}% {:>7.0}% {:>7.0}% {:>6.1}ms",
                fixture.name,
                r_oov * 100.0,
                r_pmi * 100.0,
                r_combined * 100.0,
                r_scanner * 100.0,
                elapsed.as_secs_f64() * 1000.0
            );

            // Show what each approach missed
            let combined_missed: Vec<&String> = expected
                .iter()
                .filter(|exp| !combined.iter().any(|ext| fuzzy_match(ext, exp)))
                .collect();
            let scanner_missed: Vec<&String> = expected
                .iter()
                .filter(|exp| !scanner_names.iter().any(|ext| fuzzy_match(ext, exp)))
                .collect();

            // Show entities found by OOV+PMI but missed by scanner (and vice versa)
            let oov_only: Vec<&String> = expected
                .iter()
                .filter(|exp| {
                    combined.iter().any(|ext| fuzzy_match(ext, exp))
                        && !scanner_names.iter().any(|ext| fuzzy_match(ext, exp))
                })
                .collect();
            let scanner_only: Vec<&String> = expected
                .iter()
                .filter(|exp| {
                    scanner_names.iter().any(|ext| fuzzy_match(ext, exp))
                        && !combined.iter().any(|ext| fuzzy_match(ext, exp))
                })
                .collect();

            if !oov_only.is_empty() {
                eprintln!(
                    "  OOV+PMI finds, scanner misses: {}",
                    oov_only
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if !scanner_only.is_empty() {
                eprintln!(
                    "  Scanner finds, OOV+PMI misses: {}",
                    scanner_only
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if !combined_missed.is_empty() {
                eprintln!(
                    "  Both miss: {}",
                    combined_missed
                        .iter()
                        .filter(|exp| scanner_missed.contains(exp))
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            eprintln!(
                "  Candidates: OOV={} + {} runs, PMI={} bigrams, Scanner={}",
                oov_singles.len(),
                oov_runs.len(),
                pmi_candidates.len(),
                scanner_results.len()
            );

            totals[0] += r_oov;
            totals[1] += r_pmi;
            totals[2] += r_combined;
            totals[3] += r_scanner;
            fixture_count += 1;
        }

        let n = fixture_count as f64;
        eprintln!("{}", "─".repeat(75));
        eprintln!(
            "{:<25} {:>7.0}% {:>7.0}% {:>7.0}% {:>7.0}%",
            "AVERAGE",
            totals[0] / n * 100.0,
            totals[1] / n * 100.0,
            totals[2] / n * 100.0,
            totals[3] / n * 100.0
        );
        eprintln!("═══════════════════════════════════════════════════════════════════════════");
        eprintln!("PMI threshold: {:.1}", pmi_threshold);
        eprintln!("Dictionary: en_US Hunspell (~49k entries)");
        eprintln!("Cost: ZERO LLM calls | sub-second per fixture");
        eprintln!("═══════════════════════════════════════════════════════════════════════════");

        // === Union analysis: what does combining BOTH approaches yield? ===
        eprintln!("\n── UNION ANALYSIS: OOV+PMI ∪ Scanner ──────────────────────────────────");
        let mut total_union_recall = 0.0;
        let mut total_expected = 0;
        let mut total_union_found = 0;
        for fixture in &fixture_list {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => continue,
            };
            let text = std::fs::read_to_string(fixture.path).unwrap();

            let oov_singles = extract_oov_singles(&text, &dict, &stops);
            let oov_runs = extract_oov_runs(&text, &dict, &stops);
            let pmi_bigrams = compute_pmi_bigrams(&text, pmi_threshold);
            let scanner_results = scan_proper_nouns(&text, &[]);

            let mut union: HashSet<String> = HashSet::new();
            for s in &oov_singles {
                union.insert(s.to_lowercase());
            }
            for s in &oov_runs {
                union.insert(s.to_lowercase());
            }
            for (s, _) in &pmi_bigrams {
                union.insert(s.clone());
            }
            for e in &scanner_results {
                union.insert(e.name.to_lowercase());
            }

            let union_vec: Vec<String> = union.into_iter().collect();
            let (found, r) = recall(&union_vec, expected);
            total_union_recall += r;
            total_union_found += found;
            total_expected += expected.len();

            eprintln!(
                "{:<25} {:>7.0}% ({}/{})",
                fixture.name,
                r * 100.0,
                found,
                expected.len()
            );
        }
        eprintln!("{}", "─".repeat(75));
        eprintln!(
            "{:<25} {:>7.0}% ({}/{})",
            "AVERAGE (union)",
            total_union_recall / fixture_count as f64 * 100.0,
            total_union_found,
            total_expected
        );
        eprintln!("═══════════════════════════════════════════════════════════════════════════");
    }

    // ════════════════════════════════════════════════════════════════════════
    // INCREMENTAL PIPELINE — Stage-by-stage benchmark
    // ════════════════════════════════════════════════════════════════════════

    // ── Stage 1: NPMI (Normalized PMI) ──────────────────────────────────────

    /// Compute NPMI for bigrams. NPMI = PMI / -log2(P(x,y)), normalized to [-1, 1].
    /// min_freq: both tokens must appear at least this many times.
    fn compute_npmi_bigrams(text: &str, threshold: f64, min_freq: usize) -> Vec<(String, f64)> {
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
            // Frequency floor: both tokens must appear min_freq times
            let f1 = *unigram_freq.get(w1).unwrap_or(&0);
            let f2 = *unigram_freq.get(w2).unwrap_or(&0);
            if f1 < min_freq || f2 < min_freq {
                continue;
            }

            let p_w1 = f1 as f64 / n_f;
            let p_w2 = f2 as f64 / n_f;
            let p_bigram = *freq as f64 / (n - 1) as f64;

            if p_bigram > 0.0 && p_w1 > 0.0 && p_w2 > 0.0 {
                let pmi = (p_bigram / (p_w1 * p_w2)).log2();
                let neg_log_p = -(p_bigram.log2());
                let npmi = if neg_log_p > 0.0 {
                    pmi / neg_log_p
                } else {
                    0.0
                };
                if npmi >= threshold {
                    results.push((format!("{} {}", w1, w2), npmi));
                }
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    // ── Stage 3: Entropy gate (reimplemented — pub(crate) not accessible) ───

    fn shannon_entropy(s: &str) -> f64 {
        let mut freq: HashMap<char, usize> = HashMap::new();
        let mut total = 0usize;
        for c in s.chars() {
            *freq.entry(c).or_default() += 1;
            total += 1;
        }
        if total == 0 {
            return 0.0;
        }
        let n = total as f64;
        freq.values()
            .map(|&f| {
                let p = f as f64 / n;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Matches resolver.rs entropy gate: min 6 chars, min 2 tokens, min 1.5 bits
    fn entropy_gate(candidate: &str) -> bool {
        let trimmed = candidate.trim();
        if trimmed.len() < 6 {
            return false;
        }
        let token_count = trimmed.split_whitespace().count();
        if token_count < 2 {
            // Single tokens: need ≥ 6 chars AND ≥ 1.5 entropy
            return shannon_entropy(trimmed) >= 1.5;
        }
        // Multi-token: need ≥ 1.5 entropy
        shannon_entropy(trimmed) >= 1.5
    }

    // ── Stage 4: Negative pattern exclusion ─────────────────────────────────

    const DETERMINERS: &[&str] = &[
        "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its",
        "our", "their", "some", "any", "no", "each", "every",
    ];

    const PREPOSITIONS: &[&str] = &[
        "in", "on", "at", "to", "for", "by", "from", "with", "of", "about", "into", "through",
        "during", "before", "after", "between", "among", "under", "above", "below", "near",
        "against", "within", "without",
    ];

    const AUX_VERBS: &[&str] = &[
        "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    ];

    const PRONOUNS: &[&str] = &[
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "who",
        "whom", "what",
    ];

    const CONJUNCTIONS: &[&str] = &["and", "or", "but", "nor", "yet", "so"];

    /// Returns (passes, rejection_reason)
    fn negative_pattern_check(candidate: &str) -> (bool, &'static str) {
        let words: Vec<&str> = candidate.split_whitespace().collect();
        if words.is_empty() {
            return (false, "empty");
        }

        let first = words[0].to_lowercase();
        let last = words[words.len() - 1].to_lowercase();

        // Starts with determiner
        if DETERMINERS.contains(&first.as_str()) {
            return (false, "starts_with_det");
        }

        // Starts with preposition
        if PREPOSITIONS.contains(&first.as_str()) {
            return (false, "starts_with_prep");
        }

        // Contains auxiliary verb
        for w in &words {
            let lower = w.to_lowercase();
            if AUX_VERBS.contains(&lower.as_str()) {
                return (false, "contains_aux_verb");
            }
        }

        // Contains pronoun
        for w in &words {
            let lower = w.to_lowercase();
            if PRONOUNS.contains(&lower.as_str()) {
                return (false, "contains_pronoun");
            }
        }

        // Starts or ends with conjunction
        if CONJUNCTIONS.contains(&first.as_str()) || CONJUNCTIONS.contains(&last.as_str()) {
            return (false, "conjunction_boundary");
        }

        // Ends with preposition
        if PREPOSITIONS.contains(&last.as_str()) {
            return (false, "ends_with_prep");
        }

        (true, "")
    }

    // ── Stage 5: Word-shape POS proxy ───────────────────────────────────────

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum PosShape {
        ProperNoun, // Title Case or ALL CAPS
        Noun,       // lowercase, no verbal suffix
        Adjective,  // -al, -ive, -ous, -ful, -ic, -able
        Verbal,     // -ing (not in exceptions), -ed, -ize, -ify
        Adverb,     // -ly
        Function,   // known function word
        Number,     // digit-containing
    }

    // Words ending -ing that are NOT verbs (proper nouns, nouns, adjectives)
    const ING_EXCEPTIONS: &[&str] = &[
        "beijing",
        "ring",
        "spring",
        "king",
        "string",
        "thing",
        "nothing",
        "something",
        "everything",
        "anything",
        "morning",
        "evening",
        "building",
        "meeting",
        "ceiling",
        "feeling",
        "sterling",
        "downing",
        "ling",
        "darling",
        "sibling",
        "offspring",
        "fling",
        "filing",
        "mining",
        "dining",
        "timing",
        "pricing",
        "trading",
        "reading",
        "leading",
        "funding",
        "banking",
        "marketing",
        "engineering",
        "publishing",
        "consulting",
        "computing",
        "manufacturing",
        "streaming",
        "processing",
    ];

    // Words ending -ly that are NOT adverbs (nouns, proper nouns, adjectives)
    const LY_EXCEPTIONS: &[&str] = &[
        "supply",
        "family",
        "assembly",
        "rally",
        "ally",
        "italy",
        "july",
        "daily",
        "weekly",
        "monthly",
        "yearly",
        "quarterly",
        "holy",
        "belly",
        "bully",
        "folly",
        "jelly",
        "jolly",
        "lily",
        "tally",
        "anomaly",
        "monopoly",
        "butterfly",
        "fly",
    ];

    fn classify_word_shape(word: &str) -> PosShape {
        let lower = word.to_lowercase();

        // Function words (highest priority — these are never entities)
        if DETERMINERS.contains(&lower.as_str())
            || PREPOSITIONS.contains(&lower.as_str())
            || AUX_VERBS.contains(&lower.as_str())
            || PRONOUNS.contains(&lower.as_str())
            || CONJUNCTIONS.contains(&lower.as_str())
        {
            return PosShape::Function;
        }

        // Numbers
        if word.chars().any(|c| c.is_numeric()) {
            return PosShape::Number;
        }

        // Title Case or ALL CAPS → likely proper noun
        let first_char = word.chars().next().unwrap_or('a');
        if first_char.is_uppercase() {
            return PosShape::ProperNoun;
        }

        // Suffix-based classification (lowercase words only)
        if lower.ends_with("ing") && lower.len() > 4 && !ING_EXCEPTIONS.contains(&lower.as_str()) {
            return PosShape::Verbal;
        }
        if lower.ends_with("ed") && lower.len() > 3 {
            return PosShape::Verbal;
        }
        if lower.ends_with("ize") || lower.ends_with("ify") {
            return PosShape::Verbal;
        }
        if lower.ends_with("ly") && lower.len() > 3 && !LY_EXCEPTIONS.contains(&lower.as_str()) {
            return PosShape::Adverb;
        }
        if lower.ends_with("al")
            || lower.ends_with("ive")
            || lower.ends_with("ous")
            || lower.ends_with("ful")
            || lower.ends_with("ic")
            || lower.ends_with("able")
            || lower.ends_with("ible")
        {
            return PosShape::Adjective;
        }

        PosShape::Noun // default: lowercase word with no special suffix
    }

    /// Returns (passes, rejection_reason)
    fn pos_shape_check(candidate: &str) -> (bool, String) {
        let words: Vec<&str> = candidate.split_whitespace().collect();
        if words.is_empty() {
            return (false, "empty".to_string());
        }

        let shapes: Vec<PosShape> = words.iter().map(|w| classify_word_shape(w)).collect();

        // Accept: all tokens are nominal (ProperNoun, Noun, Adjective, Number)
        // Reject: any token is Verbal, Adverb, Function
        for (i, shape) in shapes.iter().enumerate() {
            match shape {
                PosShape::Verbal => {
                    return (
                        false,
                        format!("verbal:'{}' ({})", words[i], words[i].to_lowercase()),
                    );
                }
                PosShape::Adverb => {
                    return (false, format!("adverb:'{}'", words[i]));
                }
                PosShape::Function => {
                    return (false, format!("function:'{}'", words[i]));
                }
                _ => {} // ProperNoun, Noun, Adjective, Number, Other → keep
            }
        }

        (true, String::new())
    }

    // ── Incremental pipeline test ───────────────────────────────────────────

    struct StageResult {
        #[allow(dead_code)]
        name: &'static str,
        candidates: Vec<String>,
        recall: f64,
        found: usize,
        rejected_log: Vec<(String, String)>, // (candidate, reason)
    }

    fn run_stage(
        name: &'static str,
        input: &[String],
        filter: impl Fn(&str) -> (bool, String),
        expected: &[String],
    ) -> StageResult {
        let mut kept = Vec::new();
        let mut rejected_log = Vec::new();

        for candidate in input {
            let (passes, reason) = filter(candidate);
            if passes {
                kept.push(candidate.clone());
            } else {
                rejected_log.push((candidate.clone(), reason));
            }
        }

        let (found, r) = recall(&kept, expected);
        StageResult {
            name,
            candidates: kept,
            recall: r,
            found,
            rejected_log,
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // RALPH LOOP 1: Raw PMI → NegPat → POS (skip NPMI — it kills recall)
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn spike_ralph_loop_1_skip_npmi() {
        let dict = load_dictionary();
        let gt = load_ground_truth();
        let stops = load_stop_words();
        let fixture_list = all_fixtures();

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("RALPH LOOP 1: Raw PMI → NegPat → POS (skip NPMI)");
        eprintln!("Hypothesis: NegPat + POS cut false positives without destroying recall");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<22} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Domain", "RawPMI", "+NegPat", "+POS", "Union+Scn", "Scanner"
        );
        eprintln!("{}", "─".repeat(72));

        let mut totals_r = vec![0.0f64; 5];
        let mut totals_c = vec![0usize; 5];
        let mut fixture_count = 0;

        for fixture in &fixture_list {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => {
                    continue;
                }
            };
            let text = std::fs::read_to_string(fixture.path).unwrap();

            // Raw PMI
            let raw = compute_pmi_bigrams(&text, 2.0);
            let raw_c: Vec<String> = raw.iter().map(|(s, _)| s.clone()).collect();
            let (_, raw_r) = recall(&raw_c, expected);

            // → NegPat
            let s_neg = run_stage(
                "NegPat",
                &raw_c,
                |c| {
                    let (p, r) = negative_pattern_check(c);
                    (p, r.to_string())
                },
                expected,
            );

            // → POS
            let s_pos = run_stage("POS", &s_neg.candidates, |c| pos_shape_check(c), expected);

            // Union: filtered PMI + OOV + scanner
            let oov_singles = extract_oov_singles(&text, &dict, &stops);
            let oov_runs = extract_oov_runs(&text, &dict, &stops);
            let scanner_results = scan_proper_nouns(&text, &[]);
            let scanner_names: Vec<String> = scanner_results
                .iter()
                .map(|e| e.name.to_lowercase())
                .collect();

            let mut union_set: HashSet<String> = HashSet::new();
            for s in &s_pos.candidates {
                union_set.insert(s.clone());
            }
            for s in &oov_singles {
                union_set.insert(s.to_lowercase());
            }
            for s in &oov_runs {
                union_set.insert(s.to_lowercase());
            }
            for s in &scanner_names {
                union_set.insert(s.clone());
            }
            let union_vec: Vec<String> = union_set.into_iter().collect();
            let (_, union_r) = recall(&union_vec, expected);
            let (_, scanner_r) = recall(&scanner_names, expected);

            eprintln!(
                "{:<22} {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3})",
                fixture.name,
                format!("{:.0}%", raw_r * 100.0),
                raw_c.len(),
                format!("{:.0}%", s_neg.recall * 100.0),
                s_neg.candidates.len(),
                format!("{:.0}%", s_pos.recall * 100.0),
                s_pos.candidates.len(),
                format!("{:.0}%", union_r * 100.0),
                union_vec.len(),
                format!("{:.0}%", scanner_r * 100.0),
                scanner_names.len(),
            );

            // Log entities lost at each stage
            let neg_lost: Vec<&String> = expected
                .iter()
                .filter(|exp| {
                    raw_c.iter().any(|ext| fuzzy_match(ext, exp))
                        && !s_neg.candidates.iter().any(|ext| fuzzy_match(ext, exp))
                })
                .collect();
            let pos_lost: Vec<&String> = expected
                .iter()
                .filter(|exp| {
                    s_neg.candidates.iter().any(|ext| fuzzy_match(ext, exp))
                        && !s_pos.candidates.iter().any(|ext| fuzzy_match(ext, exp))
                })
                .collect();

            if !neg_lost.is_empty() {
                eprintln!(
                    "  ⚠ NegPat lost: {}",
                    neg_lost
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if !pos_lost.is_empty() {
                eprintln!(
                    "  ⚠ POS lost: {}",
                    pos_lost
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            // Sample what POS kept (for inspection)
            if s_pos.candidates.len() <= 20 {
                eprintln!("  POS kept: {}", s_pos.candidates.join(", "));
            } else {
                let sample: Vec<&str> = s_pos
                    .candidates
                    .iter()
                    .take(15)
                    .map(|s| s.as_str())
                    .collect();
                eprintln!(
                    "  POS kept({}): {} ...",
                    s_pos.candidates.len(),
                    sample.join(", ")
                );
            }

            totals_r[0] += raw_r;
            totals_c[0] += raw_c.len();
            totals_r[1] += s_neg.recall;
            totals_c[1] += s_neg.candidates.len();
            totals_r[2] += s_pos.recall;
            totals_c[2] += s_pos.candidates.len();
            totals_r[3] += union_r;
            totals_c[3] += union_vec.len();
            totals_r[4] += scanner_r;
            totals_c[4] += scanner_names.len();
            fixture_count += 1;
        }

        let n = fixture_count as f64;
        eprintln!("{}", "─".repeat(72));
        eprintln!(
            "{:<22} {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3})",
            "AVERAGE",
            format!("{:.0}%", totals_r[0] / n * 100.0),
            totals_c[0] / fixture_count,
            format!("{:.0}%", totals_r[1] / n * 100.0),
            totals_c[1] / fixture_count,
            format!("{:.0}%", totals_r[2] / n * 100.0),
            totals_c[2] / fixture_count,
            format!("{:.0}%", totals_r[3] / n * 100.0),
            totals_c[3] / fixture_count,
            format!("{:.0}%", totals_r[4] / n * 100.0),
            totals_c[4] / fixture_count,
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════"
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    // RALPH LOOP 3: Language-agnostic filters ONLY (no hardcoded English)
    // Uses: stop-words crate + zspell dictionary + apostrophe detection
    // Zero English word lists. Zero suffix rules. Zero POS proxy.
    // ════════════════════════════════════════════════════════════════════════

    /// Language-agnostic filter: discard if ALL words are stop-words
    fn all_words_are_stop_words(candidate: &str, stops: &HashSet<String>) -> bool {
        candidate
            .split_whitespace()
            .all(|w| stops.contains(&w.to_lowercase()))
    }

    /// Language-agnostic filter: contains apostrophe (contractions/possessives)
    fn contains_apostrophe(candidate: &str) -> bool {
        candidate.contains('\'') || candidate.contains('\u{2019}') // ASCII + curly
    }

    /// Language-agnostic signal: at least one word is OOV or Title Case
    fn has_entity_signal_agnostic(
        orig_candidate: &str,
        dict: &zspell::Dictionary,
        stops: &HashSet<String>,
    ) -> bool {
        for word in orig_candidate.split_whitespace() {
            if word.len() < 2 {
                continue;
            }
            // Title Case
            if word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
            {
                return true;
            }
            // OOV
            if is_oov(word, dict, stops) {
                return true;
            }
        }
        false
    }

    #[test]
    fn spike_ralph_loop_3_lang_agnostic() {
        let dict = load_dictionary();
        let gt = load_ground_truth();
        let stops = load_stop_words();
        let fixture_list = all_fixtures();

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("RALPH LOOP 3: Language-Agnostic Filters Only");
        eprintln!("Pipeline: Raw PMI → stop-words filter → apostrophe filter → entity signal");
        eprintln!("Zero English word lists. Zero suffix rules. Zero POS proxy.");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<22} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Domain", "RawPMI", "-Stops", "-Apost", "+Signal", "Union+Scn", "Scanner"
        );
        eprintln!("{}", "─".repeat(82));

        let mut totals_r = vec![0.0f64; 6];
        let mut totals_c = vec![0usize; 6];
        let mut fixture_count = 0;

        for fixture in &fixture_list {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => {
                    continue;
                }
            };
            let text = std::fs::read_to_string(fixture.path).unwrap();

            // Original-case bigram lookup (PMI candidates are lowercase)
            let orig_words: Vec<&str> = text.unicode_words().filter(|w| w.len() >= 2).collect();
            let mut orig_bigrams: HashMap<String, String> = HashMap::new();
            for pair in orig_words.windows(2) {
                let key = format!("{} {}", pair[0].to_lowercase(), pair[1].to_lowercase());
                orig_bigrams
                    .entry(key)
                    .or_insert_with(|| format!("{} {}", pair[0], pair[1]));
            }

            // Stage 0: Raw PMI
            let raw = compute_pmi_bigrams(&text, 2.0);
            let raw_c: Vec<String> = raw.iter().map(|(s, _)| s.clone()).collect();
            let (_, raw_r) = recall(&raw_c, expected);

            // Stage 1: Remove all-stop-word bigrams
            let s1 = run_stage(
                "Stops",
                &raw_c,
                |c| {
                    if all_words_are_stop_words(c, &stops) {
                        (false, "all_stop_words".to_string())
                    } else {
                        (true, String::new())
                    }
                },
                expected,
            );

            // Stage 2: Remove candidates with apostrophes
            let s2 = run_stage(
                "Apost",
                &s1.candidates,
                |c| {
                    // Check original-case version too
                    let orig = orig_bigrams.get(c).map(|s| s.as_str()).unwrap_or(c);
                    if contains_apostrophe(c) || contains_apostrophe(orig) {
                        (false, "apostrophe".to_string())
                    } else {
                        (true, String::new())
                    }
                },
                expected,
            );

            // Stage 3: Entity signal — at least one word is Title Case or OOV
            let s3 = run_stage(
                "Signal",
                &s2.candidates,
                |c| {
                    let orig = orig_bigrams.get(c).map(|s| s.as_str()).unwrap_or(c);
                    if has_entity_signal_agnostic(orig, &dict, &stops) {
                        (true, String::new())
                    } else {
                        (false, "no_signal".to_string())
                    }
                },
                expected,
            );

            // Union: filtered PMI + OOV singles/runs + scanner
            let oov_singles = extract_oov_singles(&text, &dict, &stops);
            let oov_runs = extract_oov_runs(&text, &dict, &stops);
            let scanner_results = scan_proper_nouns(&text, &[]);
            let scanner_names: Vec<String> = scanner_results
                .iter()
                .map(|e| e.name.to_lowercase())
                .collect();

            let mut union_set: HashSet<String> = HashSet::new();
            for s in &s3.candidates {
                union_set.insert(s.clone());
            }
            for s in &oov_singles {
                union_set.insert(s.to_lowercase());
            }
            for s in &oov_runs {
                union_set.insert(s.to_lowercase());
            }
            for s in &scanner_names {
                union_set.insert(s.clone());
            }
            let union_vec: Vec<String> = union_set.into_iter().collect();
            let (_, union_r) = recall(&union_vec, expected);
            let (_, scanner_r) = recall(&scanner_names, expected);

            eprintln!(
                "{:<22} {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3})",
                fixture.name,
                format!("{:.0}%", raw_r * 100.0), raw_c.len(),
                format!("{:.0}%", s1.recall * 100.0), s1.candidates.len(),
                format!("{:.0}%", s2.recall * 100.0), s2.candidates.len(),
                format!("{:.0}%", s3.recall * 100.0), s3.candidates.len(),
                format!("{:.0}%", union_r * 100.0), union_vec.len(),
                format!("{:.0}%", scanner_r * 100.0), scanner_names.len(),
            );

            // Log entities lost at each stage
            for (stage_name, stage_cands, prev_cands) in [
                ("Stops", &s1.candidates, &raw_c),
                ("Apost", &s2.candidates, &s1.candidates),
                ("Signal", &s3.candidates, &s2.candidates),
            ] {
                let lost: Vec<&String> = expected
                    .iter()
                    .filter(|exp| {
                        prev_cands.iter().any(|ext| fuzzy_match(ext, exp))
                            && !stage_cands.iter().any(|ext| fuzzy_match(ext, exp))
                    })
                    .collect();
                if !lost.is_empty() {
                    eprintln!(
                        "  {} lost: {}",
                        stage_name,
                        lost.iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }

            // Show what survived
            if s3.candidates.len() <= 20 {
                eprintln!("  Survived: {}", s3.candidates.join(", "));
            } else {
                let sample: Vec<&str> = s3.candidates.iter().take(15).map(|s| s.as_str()).collect();
                eprintln!(
                    "  Survived({}): {} ...",
                    s3.candidates.len(),
                    sample.join(", ")
                );
            }

            totals_r[0] += raw_r;
            totals_c[0] += raw_c.len();
            totals_r[1] += s1.recall;
            totals_c[1] += s1.candidates.len();
            totals_r[2] += s2.recall;
            totals_c[2] += s2.candidates.len();
            totals_r[3] += s3.recall;
            totals_c[3] += s3.candidates.len();
            totals_r[4] += union_r;
            totals_c[4] += union_vec.len();
            totals_r[5] += scanner_r;
            totals_c[5] += scanner_names.len();
            fixture_count += 1;
        }

        let n = fixture_count as f64;
        eprintln!("{}", "─".repeat(82));
        eprintln!(
            "{:<22} {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3})",
            "AVERAGE",
            format!("{:.0}%", totals_r[0] / n * 100.0),
            totals_c[0] / fixture_count,
            format!("{:.0}%", totals_r[1] / n * 100.0),
            totals_c[1] / fixture_count,
            format!("{:.0}%", totals_r[2] / n * 100.0),
            totals_c[2] / fixture_count,
            format!("{:.0}%", totals_r[3] / n * 100.0),
            totals_c[3] / fixture_count,
            format!("{:.0}%", totals_r[4] / n * 100.0),
            totals_c[4] / fixture_count,
            format!("{:.0}%", totals_r[5] / n * 100.0),
            totals_c[5] / fixture_count,
        );
        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("Language-agnostic: stop-words crate + zspell + apostrophe detection only");
        eprintln!("No English word lists. No POS. No suffix rules.");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════"
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    // RALPH LOOP 2: Raw PMI → NegPat → POS → Entity Signal filter
    // Fix: -ly exceptions (supply, family). New: require Title Case or OOV.
    // ════════════════════════════════════════════════════════════════════════

    /// Common English contractions that are OOV but not entity signals
    const CONTRACTIONS: &[&str] = &[
        "i'm",
        "i'll",
        "i've",
        "i'd",
        "we're",
        "we'll",
        "we've",
        "we'd",
        "you're",
        "you'll",
        "you've",
        "you'd",
        "they're",
        "they'll",
        "they've",
        "they'd",
        "he's",
        "he'll",
        "he'd",
        "she's",
        "she'll",
        "she'd",
        "it's",
        "it'll",
        "that's",
        "there's",
        "here's",
        "what's",
        "who's",
        "how's",
        "where's",
        "let's",
        "can't",
        "won't",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "hasn't",
        "haven't",
        "hadn't",
        "wouldn't",
        "shouldn't",
        "couldn't",
        "mustn't",
    ];

    /// Entity signal: at least one word must be Title Case or OOV.
    /// Contractions and very short words don't count as signals.
    fn has_entity_signal(
        candidate: &str,
        dict: &zspell::Dictionary,
        stops: &HashSet<String>,
    ) -> (bool, String) {
        let words: Vec<&str> = candidate.split_whitespace().collect();

        // Reject if ANY word is a contraction (strong non-entity signal)
        for word in &words {
            if CONTRACTIONS.contains(&word.to_lowercase().as_str()) {
                return (false, format!("contraction:'{}'", word));
            }
        }

        for word in &words {
            // Skip very short words as entity signals (< 3 chars)
            if word.len() < 3 {
                continue;
            }

            // Title Case check (original casing preserved in some candidates)
            if word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
            {
                return (true, format!("title_case:'{}'", word));
            }
            // OOV check
            if is_oov(word, dict, stops) {
                return (true, format!("oov:'{}'", word));
            }
        }
        (false, "no_signal".to_string())
    }

    #[test]
    fn spike_ralph_loop_2_entity_signal() {
        let dict = load_dictionary();
        let gt = load_ground_truth();
        let stops = load_stop_words();
        let fixture_list = all_fixtures();

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("RALPH LOOP 2: Raw PMI → NegPat → POS → Entity Signal (Title Case | OOV)");
        eprintln!("Fix: -ly exceptions. New: at least one word must be Title Case or OOV");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<22} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Domain", "RawPMI", "+NegPat", "+POS", "+Signal", "Union+Scn", "Scanner"
        );
        eprintln!("{}", "─".repeat(82));

        let mut totals_r = vec![0.0f64; 6];
        let mut totals_c = vec![0usize; 6];
        let mut fixture_count = 0;

        for fixture in &fixture_list {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => {
                    continue;
                }
            };
            let text = std::fs::read_to_string(fixture.path).unwrap();

            // Raw PMI (candidates are lowercase — need original text for Title Case check)
            let raw = compute_pmi_bigrams(&text, 2.0);
            let raw_c: Vec<String> = raw.iter().map(|(s, _)| s.clone()).collect();
            let (_, raw_r) = recall(&raw_c, expected);

            // We need the original-case bigrams for Title Case detection
            let orig_words: Vec<&str> = text.unicode_words().filter(|w| w.len() >= 2).collect();
            let mut orig_bigrams: HashMap<String, String> = HashMap::new();
            for pair in orig_words.windows(2) {
                let key = format!("{} {}", pair[0].to_lowercase(), pair[1].to_lowercase());
                // Keep the original-case version
                orig_bigrams
                    .entry(key)
                    .or_insert_with(|| format!("{} {}", pair[0], pair[1]));
            }

            // → NegPat (on lowercase candidates)
            let s_neg = run_stage(
                "NegPat",
                &raw_c,
                |c| {
                    let (p, r) = negative_pattern_check(c);
                    (p, r.to_string())
                },
                expected,
            );

            // → POS (on lowercase candidates — POS proxy checks suffixes)
            let s_pos = run_stage("POS", &s_neg.candidates, |c| pos_shape_check(c), expected);

            // → Entity Signal: check original-case version for Title Case, or OOV
            let s_sig = run_stage(
                "Signal",
                &s_pos.candidates,
                |c| {
                    // Look up the original-case version
                    let orig = orig_bigrams.get(c).map(|s| s.as_str()).unwrap_or(c);
                    has_entity_signal(orig, &dict, &stops)
                },
                expected,
            );

            // Union: filtered PMI + OOV + scanner
            let oov_singles = extract_oov_singles(&text, &dict, &stops);
            let oov_runs = extract_oov_runs(&text, &dict, &stops);
            let scanner_results = scan_proper_nouns(&text, &[]);
            let scanner_names: Vec<String> = scanner_results
                .iter()
                .map(|e| e.name.to_lowercase())
                .collect();

            let mut union_set: HashSet<String> = HashSet::new();
            for s in &s_sig.candidates {
                union_set.insert(s.clone());
            }
            for s in &oov_singles {
                union_set.insert(s.to_lowercase());
            }
            for s in &oov_runs {
                union_set.insert(s.to_lowercase());
            }
            for s in &scanner_names {
                union_set.insert(s.clone());
            }
            let union_vec: Vec<String> = union_set.into_iter().collect();
            let (_, union_r) = recall(&union_vec, expected);
            let (_, scanner_r) = recall(&scanner_names, expected);

            eprintln!(
                "{:<22} {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3})",
                fixture.name,
                format!("{:.0}%", raw_r * 100.0), raw_c.len(),
                format!("{:.0}%", s_neg.recall * 100.0), s_neg.candidates.len(),
                format!("{:.0}%", s_pos.recall * 100.0), s_pos.candidates.len(),
                format!("{:.0}%", s_sig.recall * 100.0), s_sig.candidates.len(),
                format!("{:.0}%", union_r * 100.0), union_vec.len(),
                format!("{:.0}%", scanner_r * 100.0), scanner_names.len(),
            );

            // Log entities lost at signal stage
            let sig_lost: Vec<&String> = expected
                .iter()
                .filter(|exp| {
                    s_pos.candidates.iter().any(|ext| fuzzy_match(ext, exp))
                        && !s_sig.candidates.iter().any(|ext| fuzzy_match(ext, exp))
                })
                .collect();
            if !sig_lost.is_empty() {
                eprintln!(
                    "  ⚠ Signal lost: {}",
                    sig_lost
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            // Show what survived all filters
            if s_sig.candidates.len() <= 25 {
                eprintln!("  Survived: {}", s_sig.candidates.join(", "));
            } else {
                let sample: Vec<&str> = s_sig
                    .candidates
                    .iter()
                    .take(20)
                    .map(|s| s.as_str())
                    .collect();
                eprintln!(
                    "  Survived({}): {} ...",
                    s_sig.candidates.len(),
                    sample.join(", ")
                );
            }

            totals_r[0] += raw_r;
            totals_c[0] += raw_c.len();
            totals_r[1] += s_neg.recall;
            totals_c[1] += s_neg.candidates.len();
            totals_r[2] += s_pos.recall;
            totals_c[2] += s_pos.candidates.len();
            totals_r[3] += s_sig.recall;
            totals_c[3] += s_sig.candidates.len();
            totals_r[4] += union_r;
            totals_c[4] += union_vec.len();
            totals_r[5] += scanner_r;
            totals_c[5] += scanner_names.len();
            fixture_count += 1;
        }

        let n = fixture_count as f64;
        eprintln!("{}", "─".repeat(82));
        eprintln!(
            "{:<22} {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3})",
            "AVERAGE",
            format!("{:.0}%", totals_r[0] / n * 100.0),
            totals_c[0] / fixture_count,
            format!("{:.0}%", totals_r[1] / n * 100.0),
            totals_c[1] / fixture_count,
            format!("{:.0}%", totals_r[2] / n * 100.0),
            totals_c[2] / fixture_count,
            format!("{:.0}%", totals_r[3] / n * 100.0),
            totals_c[3] / fixture_count,
            format!("{:.0}%", totals_r[4] / n * 100.0),
            totals_c[4] / fixture_count,
            format!("{:.0}%", totals_r[5] / n * 100.0),
            totals_c[5] / fixture_count,
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("Entity Signal: candidate must have ≥1 word that is Title Case or OOV");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════"
        );
    }

    #[test]
    fn spike_incremental_pipeline() {
        let dict = load_dictionary();
        let gt = load_ground_truth();
        let stops = load_stop_words();
        let fixture_list = all_fixtures();

        let npmi_threshold = 0.0; // NPMI > 0 = co-occurrence beats random
        let min_freq = 2; // lower than literature's 10 — our fixtures are short

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SPIKE: Incremental PMI Pipeline — Stage-by-Stage Benchmark");
        eprintln!(
            "Stages: RawPMI → NPMI(>{:.1},freq≥{}) → Entropy → NegPat → POS → +Scanner",
            npmi_threshold, min_freq
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        // Column headers
        eprintln!(
            "{:<22} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Domain", "RawPMI", "NPMI", "Entropy", "NegPat", "POS", "Union+Scn", "Scanner"
        );
        eprintln!("{}", "─".repeat(102));

        let stage_count = 7; // raw, npmi, entropy, negpat, pos, union, scanner
        let mut totals_recall = vec![0.0f64; stage_count];
        let mut totals_cands = vec![0usize; stage_count];
        let mut fixture_count = 0;

        for fixture in &fixture_list {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => {
                    eprintln!("{:<22} SKIP", fixture.name);
                    continue;
                }
            };

            let text = std::fs::read_to_string(fixture.path).expect("read fixture");

            // ── Stage 0: Raw PMI baseline ──
            let raw_pmi = compute_pmi_bigrams(&text, 2.0);
            let raw_candidates: Vec<String> = raw_pmi.iter().map(|(s, _)| s.clone()).collect();
            let (raw_found, raw_recall) = recall(&raw_candidates, expected);

            // ── Stage 1+2: NPMI with frequency floor ──
            let npmi = compute_npmi_bigrams(&text, npmi_threshold, min_freq);
            let npmi_candidates: Vec<String> = npmi.iter().map(|(s, _)| s.clone()).collect();
            let (npmi_found, npmi_recall) = recall(&npmi_candidates, expected);

            // ── Stage 3: Entropy gate ──
            let s3 = run_stage(
                "Entropy",
                &npmi_candidates,
                |c| {
                    if entropy_gate(c) {
                        (true, String::new())
                    } else {
                        (false, format!("entropy={:.2}", shannon_entropy(c)))
                    }
                },
                expected,
            );

            // ── Stage 4: Negative patterns ──
            let s4 = run_stage(
                "NegPat",
                &s3.candidates,
                |c| {
                    let (pass, reason) = negative_pattern_check(c);
                    (pass, reason.to_string())
                },
                expected,
            );

            // ── Stage 5: POS shape filter ──
            let s5 = run_stage("POS", &s4.candidates, |c| pos_shape_check(c), expected);

            // ── Stage 6: Union with OOV singles + scanner ──
            let oov_singles = extract_oov_singles(&text, &dict, &stops);
            let oov_runs = extract_oov_runs(&text, &dict, &stops);
            let scanner_results = scan_proper_nouns(&text, &[]);
            let scanner_names: Vec<String> = scanner_results
                .iter()
                .map(|e| e.name.to_lowercase())
                .collect();

            let mut union_set: HashSet<String> = HashSet::new();
            for s in &s5.candidates {
                union_set.insert(s.clone());
            }
            for s in &oov_singles {
                union_set.insert(s.to_lowercase());
            }
            for s in &oov_runs {
                union_set.insert(s.to_lowercase());
            }
            for s in &scanner_names {
                union_set.insert(s.clone());
            }
            let union_vec: Vec<String> = union_set.into_iter().collect();
            let (_union_found, union_recall) = recall(&union_vec, expected);

            let (_scanner_found, scanner_recall) = recall(&scanner_names, expected);

            // ── Print summary row ──
            eprintln!(
                "{:<22} {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3})",
                fixture.name,
                format!("{:.0}%", raw_recall * 100.0), raw_candidates.len(),
                format!("{:.0}%", npmi_recall * 100.0), npmi_candidates.len(),
                format!("{:.0}%", s3.recall * 100.0), s3.candidates.len(),
                format!("{:.0}%", s4.recall * 100.0), s4.candidates.len(),
                format!("{:.0}%", s5.recall * 100.0), s5.candidates.len(),
                format!("{:.0}%", union_recall * 100.0), union_vec.len(),
                format!("{:.0}%", scanner_recall * 100.0), scanner_names.len(),
            );

            // ── Detailed rejection log (show entities lost at each stage) ──
            let stages = [
                ("NPMI", &npmi_candidates, npmi_found),
                ("Entropy", &s3.candidates, s3.found),
                ("NegPat", &s4.candidates, s4.found),
                ("POS", &s5.candidates, s5.found),
            ];
            let mut prev_found = raw_found;
            for (stage_name, _cands, found) in &stages {
                if *found < prev_found {
                    // Find which entities were lost
                    let prev_stage_cands = match *stage_name {
                        "NPMI" => &raw_candidates,
                        "Entropy" => &npmi_candidates,
                        "NegPat" => &s3.candidates,
                        "POS" => &s4.candidates,
                        _ => &raw_candidates,
                    };
                    let lost: Vec<&String> = expected
                        .iter()
                        .filter(|exp| {
                            prev_stage_cands.iter().any(|ext| fuzzy_match(ext, exp))
                                && !match *stage_name {
                                    "NPMI" => {
                                        npmi_candidates.iter().any(|ext| fuzzy_match(ext, exp))
                                    }
                                    "Entropy" => {
                                        s3.candidates.iter().any(|ext| fuzzy_match(ext, exp))
                                    }
                                    "NegPat" => {
                                        s4.candidates.iter().any(|ext| fuzzy_match(ext, exp))
                                    }
                                    "POS" => s5.candidates.iter().any(|ext| fuzzy_match(ext, exp)),
                                    _ => false,
                                }
                        })
                        .collect();
                    if !lost.is_empty() {
                        eprintln!(
                            "  ⚠ {} lost entities: {}",
                            stage_name,
                            lost.iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                }
                prev_found = *found;
            }

            // Show top rejected candidates per stage (sample for inspection)
            for (log, stage_name) in [
                (&s3.rejected_log, "Entropy"),
                (&s4.rejected_log, "NegPat"),
                (&s5.rejected_log, "POS"),
            ] {
                if !log.is_empty() {
                    let sample: Vec<String> = log
                        .iter()
                        .take(5)
                        .map(|(c, r)| format!("\"{}\"({})", c, r))
                        .collect();
                    let remaining = if log.len() > 5 {
                        format!(" +{} more", log.len() - 5)
                    } else {
                        String::new()
                    };
                    eprintln!(
                        "  {} rejected({}): {}{}",
                        stage_name,
                        log.len(),
                        sample.join(", "),
                        remaining
                    );
                }
            }

            totals_recall[0] += raw_recall;
            totals_recall[1] += npmi_recall;
            totals_recall[2] += s3.recall;
            totals_recall[3] += s4.recall;
            totals_recall[4] += s5.recall;
            totals_recall[5] += union_recall;
            totals_recall[6] += scanner_recall;
            totals_cands[0] += raw_candidates.len();
            totals_cands[1] += npmi_candidates.len();
            totals_cands[2] += s3.candidates.len();
            totals_cands[3] += s4.candidates.len();
            totals_cands[4] += s5.candidates.len();
            totals_cands[5] += union_vec.len();
            totals_cands[6] += scanner_names.len();
            fixture_count += 1;
        }

        let n = fixture_count as f64;
        eprintln!("{}", "─".repeat(102));
        eprintln!(
            "{:<22} {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3}) {:>4}({:>3})",
            "AVERAGE",
            format!("{:.0}%", totals_recall[0] / n * 100.0), totals_cands[0] / fixture_count,
            format!("{:.0}%", totals_recall[1] / n * 100.0), totals_cands[1] / fixture_count,
            format!("{:.0}%", totals_recall[2] / n * 100.0), totals_cands[2] / fixture_count,
            format!("{:.0}%", totals_recall[3] / n * 100.0), totals_cands[3] / fixture_count,
            format!("{:.0}%", totals_recall[4] / n * 100.0), totals_cands[4] / fixture_count,
            format!("{:.0}%", totals_recall[5] / n * 100.0), totals_cands[5] / fixture_count,
            format!("{:.0}%", totals_recall[6] / n * 100.0), totals_cands[6] / fixture_count,
        );
        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!(
            "NPMI threshold: {:.1} | Freq floor: {} | Entropy: 1.5 bits, 6 chars min",
            npmi_threshold, min_freq
        );
        eprintln!("Read: recall%(candidate_count) — lower count at same recall = better precision");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════"
        );
    }
}
