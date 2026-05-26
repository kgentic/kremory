use std::collections::HashSet;

use unicode_segmentation::UnicodeSegmentation;

pub trait GroundingChecker: Send + Sync {
    fn is_grounded(&self, entity_name: &str, source_text: &str) -> bool;
}

#[derive(Debug, Clone)]
pub(crate) struct TokenOverlapGroundingChecker {
    pub(crate) min_token_overlap: usize,
    pub(crate) require_head_noun: bool,
}

impl Default for TokenOverlapGroundingChecker {
    fn default() -> Self {
        Self {
            min_token_overlap: 1,
            require_head_noun: true,
        }
    }
}

impl TokenOverlapGroundingChecker {
    fn tokenize(text: &str) -> Vec<String> {
        text.unicode_words()
            .map(|word| word.to_lowercase())
            .collect()
    }
}

impl GroundingChecker for TokenOverlapGroundingChecker {
    fn is_grounded(&self, entity_name: &str, source_text: &str) -> bool {
        let entity_tokens = Self::tokenize(entity_name);
        if entity_tokens.is_empty() {
            return false;
        }

        let source_tokens: HashSet<String> = Self::tokenize(source_text).into_iter().collect();
        if source_tokens.is_empty() {
            return false;
        }

        let overlap = entity_tokens
            .iter()
            .filter(|token| source_tokens.contains(token.as_str()))
            .count();

        if overlap < self.min_token_overlap {
            return false;
        }

        if self.require_head_noun {
            if let Some(head_token) = entity_tokens.last() {
                return source_tokens.contains(head_token.as_str());
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::{GroundingChecker, TokenOverlapGroundingChecker};

    #[test]
    fn exact_name_match_is_grounded() {
        let checker = TokenOverlapGroundingChecker::default();
        assert!(checker.is_grounded("Alice Johnson", "Alice Johnson joined the meeting."));
    }

    #[test]
    fn shared_head_token_phrase_is_grounded() {
        let checker = TokenOverlapGroundingChecker::default();
        assert!(checker.is_grounded(
            "Platform Team",
            "Alice updated the platform team yesterday."
        ));
    }

    #[test]
    fn overlap_without_head_token_is_not_grounded_by_default() {
        let checker = TokenOverlapGroundingChecker::default();
        assert!(!checker.is_grounded(
            "Mercury Corporation",
            "Mercury is the closest planet to the sun."
        ));
    }

    #[test]
    fn overlap_without_head_token_can_be_allowed() {
        let checker = TokenOverlapGroundingChecker {
            min_token_overlap: 1,
            require_head_noun: false,
        };
        assert!(checker.is_grounded(
            "Mercury Corporation",
            "Mercury is the closest planet to the sun."
        ));
    }

    #[test]
    fn empty_entity_name_is_not_grounded() {
        let checker = TokenOverlapGroundingChecker::default();
        assert!(!checker.is_grounded("", "Any source text."));
    }
}
