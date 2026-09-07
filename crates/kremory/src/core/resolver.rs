use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::core::config::{EntropyConfig, MinHashConfig};
use crate::core::error::Result;
use crate::core::extraction::schemas::{ResolutionVerdictWrapper, SCHEMA_RESOLUTION_VERDICT};
use crate::core::extraction::structured::StructuredCallBuilder;
use crate::core::intelligence::{EntityResolver, ExtractedEntity, ResolutionResult};
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatProvider};
use crate::core::schema::Entity;

// ---------------------------------------------------------------------------
// Part 1: Name Normalization
// ---------------------------------------------------------------------------

/// Normalize an entity name for comparison:
/// 1. Lowercase
/// 2. Trim whitespace
/// 3. Strip non-alphanumeric/non-whitespace characters
/// 4. Collapse multiple spaces to single space
pub(crate) fn normalize_name(s: &str) -> String {
    s.to_lowercase()
        .trim()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Part 2: Shannon Entropy
// ---------------------------------------------------------------------------

/// Calculate Shannon entropy (in bits) of a string based on character frequencies.
/// Used to gate MinHash eligibility — low-entropy names skip to LLM tier.
pub(crate) fn shannon_entropy(s: &str) -> f64 {
    let len = s.len() as f64;
    if len == 0.0 {
        return 0.0;
    }

    let mut freq = HashMap::new();
    for c in s.chars() {
        *freq.entry(c).or_insert(0u32) += 1;
    }

    freq.values()
        .map(|&count| {
            let p = count as f64 / len;
            -p * p.log2()
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Part 3: MinHash / LSH
// ---------------------------------------------------------------------------

/// Generate character n-gram shingles from a string.
fn shingle(text: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        return vec![text.to_string()];
    }
    chars.windows(n).map(|w| w.iter().collect()).collect()
}

/// MinHash signature: a vector of minimum hash values, one per permutation.
pub(crate) struct MinHashSignature(pub(crate) Vec<u64>);

/// Compute a MinHash signature for a set of shingles.
pub(crate) fn compute_minhash(text: &str, config: &MinHashConfig) -> MinHashSignature {
    let shingles = shingle(&normalize_name(text), config.shingle_size);
    let mut signature = vec![u64::MAX; config.num_permutations];

    for shingle_str in &shingles {
        #[allow(clippy::needless_range_loop)]
        for perm in 0..config.num_permutations {
            let hash = hash_with_seed(shingle_str, perm as u64);
            signature[perm] = signature[perm].min(hash);
        }
    }

    MinHashSignature(signature)
}

/// Deterministic hash combining a string with a seed value.
fn hash_with_seed(s: &str, seed: u64) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    s.hash(&mut hasher);
    hasher.finish()
}

/// Estimate Jaccard similarity from two MinHash signatures using band-based LSH.
pub(crate) fn jaccard_estimate(a: &MinHashSignature, b: &MinHashSignature) -> f64 {
    assert_eq!(a.0.len(), b.0.len(), "signatures must have same length");
    let matches = a.0.iter().zip(b.0.iter()).filter(|(x, y)| x == y).count();
    matches as f64 / a.0.len() as f64
}

// ---------------------------------------------------------------------------
// Part 4: Entropy Gate
// ---------------------------------------------------------------------------

/// Check if an entity name passes the entropy gate for MinHash eligibility.
/// Names that are too short, have too few tokens, or low entropy skip MinHash
/// and go directly to LLM escalation.
pub(crate) fn entropy_gate_passes(name: &str, config: &EntropyConfig) -> bool {
    let normalized = normalize_name(name);

    if normalized.len() < config.min_name_length {
        return false;
    }

    let token_count = normalized.split_whitespace().count();
    if token_count < config.min_token_count {
        return false;
    }

    shannon_entropy(&normalized) >= config.entropy_threshold
}

// ---------------------------------------------------------------------------
// Part 5: Union-Find
// ---------------------------------------------------------------------------

/// Disjoint-set data structure with path compression and union by rank.
/// Used for bulk entity deduplication: if A=B and B=C, find(A) returns C.
pub(crate) struct UnionFind {
    parent: HashMap<String, String>,
    rank: HashMap<String, usize>,
}

impl UnionFind {
    pub(crate) fn new() -> Self {
        Self {
            parent: HashMap::new(),
            rank: HashMap::new(),
        }
    }

    /// Ensure an element exists in the set.
    pub(crate) fn make_set(&mut self, id: &str) {
        if !self.parent.contains_key(id) {
            self.parent.insert(id.to_string(), id.to_string());
            self.rank.insert(id.to_string(), 0);
        }
    }

    /// Find the canonical representative with path compression.
    pub(crate) fn find(&mut self, id: &str) -> String {
        self.make_set(id);
        if self.parent[id] == id {
            return id.to_string();
        }
        let parent = self.parent[id].clone();
        let root = self.find(&parent);
        self.parent.insert(id.to_string(), root.clone());
        root
    }

    /// Union two elements by rank.
    pub(crate) fn union(&mut self, a: &str, b: &str) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let rank_a = *self.rank.get(&ra).unwrap_or(&0);
        let rank_b = *self.rank.get(&rb).unwrap_or(&0);
        if rank_a < rank_b {
            self.parent.insert(ra, rb);
        } else if rank_a > rank_b {
            self.parent.insert(rb, ra);
        } else {
            self.parent.insert(rb, ra.clone());
            *self.rank.entry(ra).or_insert(0) += 1;
        }
    }

    /// Get all groups (canonical_id → [member_ids]).
    #[allow(dead_code)]
    pub(crate) fn groups(&mut self) -> HashMap<String, Vec<String>> {
        let ids: Vec<String> = self.parent.keys().cloned().collect();
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        for id in ids {
            let root = self.find(&id);
            groups.entry(root).or_default().push(id);
        }
        groups
    }
}

// ---------------------------------------------------------------------------
// Part 6: CascadeResolver
// ---------------------------------------------------------------------------

pub(crate) struct CascadeResolver<L: ChatProvider> {
    llm: Arc<L>,
    minhash_config: MinHashConfig,
    entropy_config: EntropyConfig,
    /// Consumer-supplied model identifier. Set by the Engine via
    /// [`with_model`](Self::with_model) at construction — NOT read off
    /// `llm.model()`. Drives capability detection + metric labels for the
    /// resolution-verdict call. `None`/empty → `PromptOnly`.
    model: Option<String>,
}

impl<L: ChatProvider> CascadeResolver<L> {
    pub(crate) fn new(
        llm: Arc<L>,
        minhash_config: MinHashConfig,
        entropy_config: EntropyConfig,
    ) -> Self {
        Self {
            llm,
            minhash_config,
            entropy_config,
            model: None,
        }
    }

    /// Set the consumer-supplied model identifier. Chainable; the
    /// Engine calls this with `self.model.clone()` at construction.
    pub(crate) fn with_model(mut self, model: Option<String>) -> Self {
        self.model = model;
        self
    }
}

/// Helper to extract entity name from Entity struct (checks properties.name first, falls back to id).
pub(crate) fn entity_name(entity: &Entity) -> &str {
    entity
        .properties
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or(&entity.id)
}

impl<L: ChatProvider> CascadeResolver<L> {
    /// The cheap deterministic tiers (Tier 1 exact-normalize + Tier 2
    /// MinHash/LSH), factored out of `resolve()` so the
    /// batched resolver can run them synchronously, no-LLM, over an entity's
    /// candidate block BEFORE deciding whether the entity is ambiguous enough
    /// to need the LLM tier at all.
    ///
    /// Returns `Some(ResolutionResult::Same)` on a deterministic hit; `None`
    /// means neither cheap tier fired and the pair would escalate to Tier 3
    /// (LLM) under `resolve()`. Never returns `Some(Different)` — the cheap
    /// tiers only ever *confirm* a match, never *rule one out*; ruling out is
    /// what Tier 3 (or the batched map-back's conservative-NEW) is for.
    ///
    /// `resolve()` calls this first and only escalates to the LLM when it
    /// returns `None` — behaviour of `resolve()` is unchanged by this
    /// refactor (DRY: both callers share one implementation of Tier 1 + 2).
    pub(crate) fn resolve_deterministic(
        &self,
        candidate: &ExtractedEntity,
        existing: &Entity,
    ) -> Option<ResolutionResult> {
        // Tier 1: Exact match after normalization
        let norm_candidate = normalize_name(&candidate.name);
        let norm_existing = normalize_name(entity_name(existing));
        if norm_candidate == norm_existing {
            return Some(ResolutionResult::Same);
        }

        // Tier 2: MinHash/LSH if entropy gate passes
        if entropy_gate_passes(&candidate.name, &self.entropy_config) {
            let sig_a = compute_minhash(&candidate.name, &self.minhash_config);
            let sig_b = compute_minhash(entity_name(existing), &self.minhash_config);
            let similarity = jaccard_estimate(&sig_a, &sig_b);
            if similarity >= self.minhash_config.jaccard_threshold {
                return Some(ResolutionResult::Same);
            }
        }

        None
    }
}

impl<L: ChatProvider> EntityResolver for CascadeResolver<L> {
    async fn resolve<'a>(
        &'a self,
        candidate: &'a ExtractedEntity,
        existing: &'a Entity,
    ) -> Result<ResolutionResult> {
        // Pass 1: cheap deterministic tiers (Tier 1 + Tier 2).
        if let Some(result) = self.resolve_deterministic(candidate, existing) {
            return Ok(result);
        }

        // Tier 3: LLM escalation
        let prompt = format!(
            "Are these two entities the same real-world thing?\n\nEntity A: \"{}\" (type: {})\nEntity B: \"{}\" (type: {})\n\nEntities are duplicates only if they refer to the same real-world object or concept. Semantically equivalent descriptive labels to named entities are treated as duplicates. Distinct but related entities are NOT duplicates.\n\nRespond with a JSON object with a single field \"verdict\" whose value is exactly one of: \"same\", \"different\", or \"uncertain\".",
            candidate.name,
            candidate.label,
            entity_name(existing),
            existing.label
        );

        let resolution_msgs = vec![
            chat_msg_system(
                "You are an entity resolution system. Determine if two entity mentions refer to the same real-world thing. Output valid JSON only.",
            ),
            chat_msg_user(prompt.as_str()),
        ];
        let verdict_value = StructuredCallBuilder::new(
            self.llm.as_ref(),
            &SCHEMA_RESOLUTION_VERDICT,
            "ResolutionVerdict",
        )
        .messages(resolution_msgs)
        .model(self.model.as_deref().unwrap_or(""))
        .call()
        .await
        .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;

        let wrapper: ResolutionVerdictWrapper =
            serde_json::from_value(verdict_value).map_err(|e| {
                crate::core::error::Error::Llm(format!(
                    "resolution verdict deserialisation failed: {e}"
                ))
            })?;

        match wrapper.verdict.trim().trim_matches('"') {
            "same" => Ok(ResolutionResult::Same),
            "different" => Ok(ResolutionResult::Different),
            _ => Ok(ResolutionResult::Different), // Uncertain treated as Different (conservative)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::{EntropyConfig, MinHashConfig};
    use crate::core::intelligence::ExtractedEntity;
    use crate::core::provider::MockChatProvider;
    use crate::core::schema::Entity;
    use chrono::Utc;

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }

    fn make_entity(id: &str, label: &str, name: &str) -> Entity {
        Entity {
            id: id.to_string(),
            label: label.to_string(),
            entity_type_id: 0,
            properties: serde_json::json!({"name": name}),
            recorded_at: Utc::now(),
            updated_at: None,
            group_id: None,
            access_count: 0,
        }
    }

    fn make_extracted(label: &str, name: &str) -> ExtractedEntity {
        ExtractedEntity {
            label: label.to_string(),
            name: name.to_string(),
            properties: serde_json::json!({"name": name}),
        }
    }

    // --- Normalization ---

    #[test]
    fn test_normalize_basic() {
        assert_eq!(normalize_name("  Hello  World  "), "hello world");
    }

    #[test]
    fn test_normalize_punctuation() {
        assert_eq!(normalize_name("Acme, Inc."), "acme inc");
    }

    #[test]
    fn test_normalize_empty() {
        assert_eq!(normalize_name(""), "");
    }

    // --- Entropy ---

    #[test]
    fn test_entropy_high_entropy_string() {
        let e = shannon_entropy("Microsoft Corporation");
        assert!(
            e > 1.5,
            "expected entropy > 1.5 for 'Microsoft Corporation', got {}",
            e
        );
    }

    #[test]
    fn test_entropy_low_entropy_string() {
        let e = shannon_entropy("aaa");
        assert!(e < 0.001, "expected entropy ≈ 0.0 for 'aaa', got {}", e);
    }

    #[test]
    fn test_entropy_empty() {
        assert_eq!(shannon_entropy(""), 0.0);
    }

    // --- Entropy Gate ---

    #[test]
    fn test_entropy_gate_short_name_fails() {
        let config = EntropyConfig {
            min_name_length: 6,
            min_token_count: 2,
            entropy_threshold: 1.5,
        };
        // "AI" is 2 chars — below min_name_length of 6
        assert!(!entropy_gate_passes("AI", &config));
    }

    #[test]
    fn test_entropy_gate_single_token_fails() {
        let config = EntropyConfig {
            min_name_length: 6,
            min_token_count: 2,
            entropy_threshold: 1.5,
        };
        // "Smith" has 5 chars (fails length) but even "Smithy" (6+ chars) would fail min_token_count=2
        assert!(!entropy_gate_passes("Smithy", &config));
    }

    #[test]
    fn test_entropy_gate_passes_valid() {
        let config = EntropyConfig {
            min_name_length: 6,
            min_token_count: 2,
            entropy_threshold: 1.5,
        };
        assert!(entropy_gate_passes("Microsoft Corporation", &config));
    }

    // --- MinHash ---

    #[test]
    fn test_minhash_identical_strings_jaccard_1() {
        let config = MinHashConfig::default();
        let a = compute_minhash("Apple Inc", &config);
        let b = compute_minhash("Apple Inc", &config);
        let j = jaccard_estimate(&a, &b);
        assert!(
            (j - 1.0).abs() < 1e-9,
            "identical strings must have Jaccard = 1.0, got {}",
            j
        );
    }

    #[test]
    fn test_minhash_similar_strings_high_jaccard() {
        let config = MinHashConfig {
            num_permutations: 128,
            shingle_size: 3,
            band_size: 4,
            jaccard_threshold: 0.9,
        };
        let a = compute_minhash("Acme Corp", &config);
        let b = compute_minhash("Acme Corporation", &config);
        let j = jaccard_estimate(&a, &b);
        // "Acme Corp" shares a meaningful number of 3-gram shingles with
        // "Acme Corporation" — the estimate should be clearly above the dissimilar
        // baseline (which is near 0) and above 0.3.
        assert!(
            j > 0.3,
            "similar strings 'Acme Corp' vs 'Acme Corporation' should have Jaccard > 0.3, got {}",
            j
        );
    }

    #[test]
    fn test_minhash_different_strings_low_jaccard() {
        let config = MinHashConfig {
            num_permutations: 128,
            shingle_size: 3,
            band_size: 4,
            jaccard_threshold: 0.9,
        };
        let a = compute_minhash("Apple Inc", &config);
        let b = compute_minhash("Microsoft Corp", &config);
        let j = jaccard_estimate(&a, &b);
        assert!(
            j < 0.5,
            "dissimilar strings 'Apple Inc' vs 'Microsoft Corp' should have Jaccard < 0.5, got {}",
            j
        );
    }

    // --- Union-Find ---

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new();
        uf.union("A", "B");
        uf.union("B", "C");
        assert_eq!(
            uf.find("A"),
            uf.find("C"),
            "A and C should have the same root after union(A,B), union(B,C)"
        );
    }

    #[test]
    fn test_union_find_groups() {
        let mut uf = UnionFind::new();
        // Group 1: A, B
        uf.union("A", "B");
        // Group 2: C, D
        uf.union("C", "D");
        // Group 3: E (standalone)
        uf.make_set("E");

        let groups = uf.groups();
        assert_eq!(groups.len(), 3, "expected 3 distinct groups");

        // Re-run on the original uf
        let root_a_orig = uf.find("A");
        let root_b_orig = uf.find("B");
        assert_eq!(root_a_orig, root_b_orig);

        // Verify C and D are in the same group
        let root_c = uf.find("C");
        let root_d = uf.find("D");
        assert_eq!(root_c, root_d);

        // Verify E is alone
        let root_e = uf.find("E");
        assert_ne!(root_e, root_a_orig);
        assert_ne!(root_e, root_c);
    }

    #[test]
    fn test_union_find_self_union() {
        let mut uf = UnionFind::new();
        uf.make_set("A");
        let root_before = uf.find("A");
        uf.union("A", "A");
        let root_after = uf.find("A");
        assert_eq!(
            root_before, root_after,
            "self-union should not change the root"
        );
    }

    // --- CascadeResolver ---

    #[test]
    fn test_cascade_tier1_exact_match() {
        // "Acme Corp" normalized == "acme corp" — should return Same without calling LLM
        let llm = Arc::new(MockChatProvider::new(HashMap::new()));
        let resolver =
            CascadeResolver::new(llm, MinHashConfig::default(), EntropyConfig::default());

        let candidate = make_extracted("Organisation", "Acme Corp");
        let existing = make_entity("acme-corp", "Organisation", "acme corp");

        let result = block_on(resolver.resolve(&candidate, &existing)).unwrap();
        assert_eq!(result, ResolutionResult::Same);
    }

    #[test]
    fn test_cascade_tier3_llm_called_for_ambiguous() {
        // "AI" is short/low-entropy → fails entropy gate → skips MinHash → calls LLM.
        // The mock returns the structured verdict JSON that StructuredCallBuilder now requires.
        let mut responses = HashMap::new();
        // The LLM prompt will contain "AI" — respond with wrapped verdict JSON.
        responses.insert("AI".to_string(), r#"{"verdict":"different"}"#.to_string());
        let llm = Arc::new(MockChatProvider::new(responses));

        let entropy_config = EntropyConfig {
            min_name_length: 6,
            min_token_count: 2,
            entropy_threshold: 1.5,
        };
        let resolver = CascadeResolver::new(llm, MinHashConfig::default(), entropy_config);

        let candidate = make_extracted("Technology", "AI");
        let existing = make_entity(
            "artificial-intelligence",
            "Technology",
            "Artificial Intelligence",
        );

        let result = block_on(resolver.resolve(&candidate, &existing)).unwrap();
        // LLM responds "different" so result should be Different
        assert_eq!(result, ResolutionResult::Different);
    }

    // ── T5.3 / Tier 3 "same" verdict ─────────────────────────────────────────

    #[test]
    fn test_cascade_tier3_llm_returns_same_verdict() {
        // Mirror of test_cascade_tier3_llm_called_for_ambiguous but with mock
        // returning "same" — verifies ResolutionResult::Same from LLM path.
        let mut responses = HashMap::new();
        // The LLM prompt will contain "AI" — respond with wrapped verdict JSON.
        responses.insert("AI".to_string(), r#"{"verdict":"same"}"#.to_string());
        let llm = Arc::new(MockChatProvider::new(responses));

        let entropy_config = EntropyConfig {
            min_name_length: 6,
            min_token_count: 2,
            entropy_threshold: 1.5,
        };
        let resolver = CascadeResolver::new(llm, MinHashConfig::default(), entropy_config);

        let candidate = make_extracted("Technology", "AI");
        let existing = make_entity(
            "artificial-intelligence",
            "Technology",
            "Artificial Intelligence",
        );

        let result = block_on(resolver.resolve(&candidate, &existing)).unwrap();
        // LLM responds "same" so result should be Same
        assert_eq!(result, ResolutionResult::Same);
    }
}
