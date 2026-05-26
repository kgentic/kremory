use crate::core::config::ContentType;
use crate::core::error::Result;
use crate::core::schema::{Entity, Fact};

// ─── Result Types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionResult {
    /// Entities refer to the same real-world thing.
    Same,
    /// Entities are distinct.
    Different,
    /// Not enough information to decide.
    Uncertain,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FactRelation {
    /// New fact is compatible with existing.
    Consistent,
    /// New fact supersedes existing (e.g., role change).
    Update,
    /// New fact conflicts with existing.
    Contradiction,
}

/// A raw extraction from text — an entity mention with optional properties.
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub label: String,
    pub name: String,
    pub properties: serde_json::Value,
}

/// A raw extraction from text — a relationship between entities.
#[derive(Debug, Clone)]
pub struct ExtractedFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// True if object refers to an entity, false if literal value.
    pub is_entity_ref: bool,
    pub confidence: f64,
}

/// Full extraction result from a text chunk.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    pub entities: Vec<ExtractedEntity>,
    pub facts: Vec<ExtractedFact>,
}

// ─── Extraction Context ───────────────────────────────────────────────────────

/// Context passed to extractors to constrain and guide extraction.
pub struct ExtractionContext<'a> {
    /// If non-empty, only entities with these labels will be extracted.
    pub allowed_entity_types: &'a [String],
    /// If non-empty, only edges with these predicates will be extracted.
    pub allowed_edge_types: &'a [String],
    /// Entities already known to the graph — helps the extractor avoid duplicating context.
    pub known_entities: &'a [ExtractedEntity],
    /// Entity labels that must never be extracted regardless of `allowed_entity_types`.
    pub excluded_entity_types: &'a [String],
    /// The structural format of the input text, used to tailor the extraction prompt.
    pub content_type: ContentType,
}

impl<'a> Default for ExtractionContext<'a> {
    fn default() -> Self {
        Self {
            allowed_entity_types: &[],
            allowed_edge_types: &[],
            known_entities: &[],
            excluded_entity_types: &[],
            content_type: ContentType::Text,
        }
    }
}

// ─── Traits ──────────────────────────────────────────────────────────────────

/// Extracts entities and facts from text.
pub trait EntityExtractor: Send + Sync {
    fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> impl std::future::Future<Output = Result<ExtractionResult>> + Send + 'a;
}

/// Determines if two entity mentions refer to the same real-world thing.
pub trait EntityResolver: Send + Sync {
    fn resolve<'a>(
        &'a self,
        candidate: &'a ExtractedEntity,
        existing: &'a Entity,
    ) -> impl std::future::Future<Output = Result<ResolutionResult>> + Send + 'a;
}

/// Classifies the relationship between a new fact and an existing fact.
pub trait ContradictionDetector: Send + Sync {
    fn classify<'a>(
        &'a self,
        new_fact: &'a ExtractedFact,
        existing: &'a Fact,
    ) -> impl std::future::Future<Output = Result<FactRelation>> + Send + 'a;
}

// ─── Mock Implementations ─────────────────────────────────────────────────────

/// Mock extractor: simple pattern-based extraction for testing the orchestration pipeline.
///
/// Recognises patterns:
///   "X works at Y"   → fact(subject=X, predicate="works_at", object=Y, is_entity_ref=true)
///   "X manages Y"    → fact(subject=X, predicate="manages",  object=Y, is_entity_ref=true)
///   "X joined Y"     → fact(subject=X, predicate="joined",   object=Y, is_entity_ref=true)
///   "X left Y"       → fact(subject=X, predicate="left",     object=Y, is_entity_ref=true)
/// Capitalised words (not part of a known pattern) are collected as Person entities.
#[cfg(any(test, feature = "test-utils"))]
pub struct MockExtractor;

#[cfg(any(test, feature = "test-utils"))]
impl EntityExtractor for MockExtractor {
    async fn extract<'a>(
        &'a self,
        text: &'a str,
        _ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        let mut entities: Vec<ExtractedEntity> = Vec::new();
        let mut facts: Vec<ExtractedFact> = Vec::new();
        let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Patterns: (predicate, regex-like split marker)
        // Each entry is (predicate, split_phrase, is_entity_ref)
        let patterns: &[(&str, &str, bool)] = &[
            ("works_at", " works at ", true),
            ("manages", " manages ", true),
            ("joined", " joined ", true),
            ("left", " left ", true),
        ];

        // Split on sentence boundaries (period or comma) then scan each clause.
        let clauses: Vec<&str> = text.split(['.', ',']).collect();

        for clause in &clauses {
            let clause = clause.trim();
            let mut matched = false;

            for (predicate, marker, is_entity_ref) in patterns {
                if let Some(pos) = clause.to_lowercase().find(marker) {
                    let subject_raw = clause[..pos].trim();
                    let object_raw = clause[pos + marker.len()..].trim();

                    // Strip leading "and " from subject (e.g. "and Bob")
                    let subject = subject_raw
                        .strip_prefix("and ")
                        .unwrap_or(subject_raw)
                        .trim()
                        .to_string();
                    let object = object_raw.to_string();

                    if !subject.is_empty() && !object.is_empty() {
                        // Collect subject as entity
                        let subject_key = subject.to_lowercase();
                        if !seen_names.contains(&subject_key) {
                            seen_names.insert(subject_key);
                            entities.push(ExtractedEntity {
                                label: "Person".to_string(),
                                name: subject.clone(),
                                properties: serde_json::json!({"name": subject}),
                            });
                        }

                        // Collect object as Organisation entity if it's an entity ref
                        if *is_entity_ref {
                            let object_key = object.to_lowercase();
                            if !seen_names.contains(&object_key) {
                                seen_names.insert(object_key);
                                entities.push(ExtractedEntity {
                                    label: "Organisation".to_string(),
                                    name: object.clone(),
                                    properties: serde_json::json!({"name": object}),
                                });
                            }
                        }

                        facts.push(ExtractedFact {
                            subject,
                            predicate: predicate.to_string(),
                            object,
                            is_entity_ref: *is_entity_ref,
                            confidence: 1.0,
                        });
                        matched = true;
                    }
                }
            }

            // If no pattern matched, scan for standalone capitalised words as Person entities.
            if !matched {
                for word in clause.split_whitespace() {
                    let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
                    if clean.len() > 1 {
                        let first_char = clean.chars().next().unwrap_or_else(|| {
                            panic!("invariant: non-empty clean string has no first char")
                        });
                        if first_char.is_uppercase() {
                            let key = clean.to_lowercase();
                            // Skip common sentence-starting words that aren't names.
                            let stop_words = [
                                "The", "A", "An", "In", "On", "At", "And", "But", "Or", "For",
                                "Of", "To", "Is", "Are", "Was", "Were", "Both", "All", "Each",
                                "That", "This", "These", "Those",
                            ];
                            if !stop_words.contains(&clean.as_str()) && !seen_names.contains(&key) {
                                seen_names.insert(key);
                                entities.push(ExtractedEntity {
                                    label: "Person".to_string(),
                                    name: clean,
                                    properties: serde_json::Value::Null,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(ExtractionResult { entities, facts })
    }
}

/// Mock resolver: exact case-insensitive name match → Same, otherwise Different.
#[cfg(any(test, feature = "test-utils"))]
pub struct MockResolver;

#[cfg(any(test, feature = "test-utils"))]
impl EntityResolver for MockResolver {
    async fn resolve<'a>(
        &'a self,
        candidate: &'a ExtractedEntity,
        existing: &'a Entity,
    ) -> Result<ResolutionResult> {
        // Check the candidate name against the entity id and the "name" property.
        let candidate_lower = candidate.name.to_lowercase();
        let id_lower = existing.id.to_lowercase();

        if candidate_lower == id_lower {
            return Ok(ResolutionResult::Same);
        }

        // Also check the "name" property if present.
        if let Some(name_val) = existing.properties.get("name") {
            if let Some(name_str) = name_val.as_str() {
                if candidate_lower == name_str.to_lowercase() {
                    return Ok(ResolutionResult::Same);
                }
            }
        }

        Ok(ResolutionResult::Different)
    }
}

/// Mock contradiction detector: predicate-based classification.
///
/// - Same subject_id + same predicate + different object → Update
/// - Same subject_id + same predicate + same object     → Consistent
/// - Otherwise                                           → Consistent
#[cfg(any(test, feature = "test-utils"))]
pub struct MockContradictionDetector;

#[cfg(any(test, feature = "test-utils"))]
impl ContradictionDetector for MockContradictionDetector {
    async fn classify<'a>(
        &'a self,
        new_fact: &'a ExtractedFact,
        existing: &'a Fact,
    ) -> Result<FactRelation> {
        let subject_matches = new_fact.subject.to_lowercase() == existing.subject_id.to_lowercase();
        let predicate_matches =
            new_fact.predicate.to_lowercase() == existing.predicate.to_lowercase();

        if !subject_matches || !predicate_matches {
            return Ok(FactRelation::Consistent);
        }

        // Same subject + same predicate: compare object.
        let new_object_lower = new_fact.object.to_lowercase();
        let existing_object = existing
            .object_id
            .as_deref()
            .or(existing.object_value.as_deref())
            .unwrap_or("")
            .to_lowercase();

        if new_object_lower == existing_object {
            Ok(FactRelation::Consistent)
        } else {
            Ok(FactRelation::Update)
        }
    }
}

// ─── Pipeline Result Types ────────────────────────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct ResolvedEntity {
    pub extracted: ExtractedEntity,
    /// Existing entity ID if resolved to Same, otherwise None.
    pub matched_id: Option<String>,
    pub resolution: ResolutionResult,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct ProcessedFact {
    pub extracted: ExtractedFact,
    pub relation: FactRelation,
    /// ID of the conflicting/superseded fact if relation is Update or Contradiction.
    pub conflicting_fact_id: Option<i64>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct ProcessedIngestion {
    pub entities: Vec<ResolvedEntity>,
    pub facts: Vec<ProcessedFact>,
}

// ─── Orchestration Pipeline ───────────────────────────────────────────────────

/// Orchestrates the intelligence pipeline: extract → resolve → contradict-check → return results.
///
/// The caller is responsible for committing the results to the graph store.
#[allow(dead_code)]
pub(crate) struct IntelligencePipeline<E, R, C> {
    extractor: E,
    resolver: R,
    contradiction_detector: C,
}

#[allow(dead_code)]
impl<E: EntityExtractor, R: EntityResolver, C: ContradictionDetector>
    IntelligencePipeline<E, R, C>
{
    pub fn new(extractor: E, resolver: R, contradiction_detector: C) -> Self {
        Self {
            extractor,
            resolver,
            contradiction_detector,
        }
    }

    /// Process a text chunk through the full pipeline:
    /// 1. Extract entities and facts from text.
    /// 2. For each extracted entity, resolve against existing graph entities.
    /// 3. For each extracted fact, check for contradictions with existing facts.
    /// 4. Return the processed results — the caller handles storage.
    pub async fn process(
        &self,
        text: &str,
        existing_entities: &[Entity],
        existing_facts: &[Fact],
    ) -> Result<ProcessedIngestion> {
        let ctx = ExtractionContext::default();
        let extraction = self.extractor.extract(text, &ctx).await?;

        // Resolve entities against the existing graph.
        let mut resolved_entities = Vec::new();
        for extracted in &extraction.entities {
            let mut resolution = ResolvedEntity {
                extracted: extracted.clone(),
                matched_id: None,
                resolution: ResolutionResult::Different,
            };
            for existing in existing_entities {
                let result = self.resolver.resolve(extracted, existing).await?;
                if result == ResolutionResult::Same {
                    resolution.matched_id = Some(existing.id.clone());
                    resolution.resolution = ResolutionResult::Same;
                    break;
                }
            }
            resolved_entities.push(resolution);
        }

        // Check each extracted fact against existing facts.
        let mut processed_facts = Vec::new();
        for new_fact in &extraction.facts {
            let mut relation = FactRelation::Consistent;
            let mut conflicting_fact_id: Option<i64> = None;
            for existing in existing_facts {
                let result = self
                    .contradiction_detector
                    .classify(new_fact, existing)
                    .await?;
                if result != FactRelation::Consistent {
                    conflicting_fact_id = Some(existing.id);
                    relation = result;
                    break;
                }
            }
            processed_facts.push(ProcessedFact {
                extracted: new_fact.clone(),
                relation,
                conflicting_fact_id,
            });
        }

        Ok(ProcessedIngestion {
            entities: resolved_entities,
            facts: processed_facts,
        })
    }
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_entity(id: &str, label: &str, name: &str) -> Entity {
        Entity {
            id: id.to_string(),
            label: label.to_string(),
            properties: serde_json::json!({"name": name}),
            recorded_at: Utc::now(),
            updated_at: None,
            group_id: None,
            access_count: 0,
        }
    }

    fn make_fact(
        id: i64,
        subject_id: &str,
        predicate: &str,
        object_id: Option<&str>,
        object_value: Option<&str>,
    ) -> Fact {
        let now = Utc::now();
        Fact {
            id,
            subject_id: subject_id.to_string(),
            predicate: predicate.to_string(),
            object_id: object_id.map(str::to_string),
            object_value: object_value.map(str::to_string),
            properties: None,
            valid_from: now,
            valid_to: None,
            recorded_at: now,
            expired_at: None,
            invalid_at: None,
            group_id: None,
            confidence: 1.0,
            source_episode_id: None,
            memory_type: None,
            content_hash: None,
            access_count: 0,
        }
    }

    #[tokio::test]
    async fn test_mock_extractor_extracts_entities_and_facts() {
        let extractor = MockExtractor;
        let ctx = ExtractionContext::default();
        let result = extractor
            .extract("Alice works at Acme Corp", &ctx)
            .await
            .unwrap();

        assert!(
            !result.entities.is_empty(),
            "should extract at least one entity"
        );
        assert!(!result.facts.is_empty(), "should extract at least one fact");

        let fact = &result.facts[0];
        assert_eq!(fact.predicate, "works_at");
        assert_eq!(fact.subject.to_lowercase(), "alice");
        assert_eq!(fact.object.to_lowercase(), "acme corp");
        assert!(fact.is_entity_ref);
    }

    #[tokio::test]
    async fn test_mock_extractor_multiple_facts() {
        let extractor = MockExtractor;
        let ctx = ExtractionContext::default();
        let result = extractor
            .extract("Alice works at Acme Corp. Bob manages Alice.", &ctx)
            .await
            .unwrap();

        assert_eq!(result.facts.len(), 2, "should extract two facts");
        let predicates: Vec<&str> = result.facts.iter().map(|f| f.predicate.as_str()).collect();
        assert!(predicates.contains(&"works_at"));
        assert!(predicates.contains(&"manages"));
    }

    #[tokio::test]
    async fn test_mock_resolver_same_name_matches() {
        let resolver = MockResolver;
        let candidate = ExtractedEntity {
            label: "Person".to_string(),
            name: "Alice".to_string(),
            properties: serde_json::Value::Null,
        };
        let existing = make_entity("alice", "Person", "Alice");

        let result = resolver.resolve(&candidate, &existing).await.unwrap();
        assert_eq!(result, ResolutionResult::Same);
    }

    #[tokio::test]
    async fn test_mock_resolver_case_insensitive_match() {
        let resolver = MockResolver;
        let candidate = ExtractedEntity {
            label: "Person".to_string(),
            name: "ALICE".to_string(),
            properties: serde_json::Value::Null,
        };
        let existing = make_entity("alice", "Person", "Alice");

        let result = resolver.resolve(&candidate, &existing).await.unwrap();
        assert_eq!(result, ResolutionResult::Same);
    }

    #[tokio::test]
    async fn test_mock_resolver_different_name_differs() {
        let resolver = MockResolver;
        let candidate = ExtractedEntity {
            label: "Person".to_string(),
            name: "Alice".to_string(),
            properties: serde_json::Value::Null,
        };
        let existing = make_entity("bob", "Person", "Bob");

        let result = resolver.resolve(&candidate, &existing).await.unwrap();
        assert_eq!(result, ResolutionResult::Different);
    }

    #[tokio::test]
    async fn test_mock_contradiction_same_predicate_different_object_is_update() {
        let detector = MockContradictionDetector;
        let new_fact = ExtractedFact {
            subject: "alice".to_string(),
            predicate: "works_at".to_string(),
            object: "newco".to_string(),
            is_entity_ref: true,
            confidence: 1.0,
        };
        let existing = make_fact(1, "alice", "works_at", Some("acme"), None);

        let result = detector.classify(&new_fact, &existing).await.unwrap();
        assert_eq!(result, FactRelation::Update);
    }

    #[tokio::test]
    async fn test_mock_contradiction_same_predicate_same_object_is_consistent() {
        let detector = MockContradictionDetector;
        let new_fact = ExtractedFact {
            subject: "alice".to_string(),
            predicate: "works_at".to_string(),
            object: "acme".to_string(),
            is_entity_ref: true,
            confidence: 1.0,
        };
        let existing = make_fact(1, "alice", "works_at", Some("acme"), None);

        let result = detector.classify(&new_fact, &existing).await.unwrap();
        assert_eq!(result, FactRelation::Consistent);
    }

    #[tokio::test]
    async fn test_mock_contradiction_different_subject_is_consistent() {
        let detector = MockContradictionDetector;
        let new_fact = ExtractedFact {
            subject: "bob".to_string(),
            predicate: "works_at".to_string(),
            object: "newco".to_string(),
            is_entity_ref: true,
            confidence: 1.0,
        };
        let existing = make_fact(1, "alice", "works_at", Some("acme"), None);

        let result = detector.classify(&new_fact, &existing).await.unwrap();
        assert_eq!(result, FactRelation::Consistent);
    }

    #[tokio::test]
    async fn test_pipeline_full_flow() {
        let pipeline =
            IntelligencePipeline::new(MockExtractor, MockResolver, MockContradictionDetector);

        let existing_entities = vec![make_entity("alice", "Person", "Alice")];
        let existing_facts = vec![make_fact(1, "alice", "works_at", Some("acme_corp"), None)];

        let result = pipeline
            .process("Alice works at NewCo", &existing_entities, &existing_facts)
            .await
            .unwrap();

        // Should have extracted entities
        assert!(
            !result.entities.is_empty(),
            "pipeline should produce entities"
        );

        // Alice should resolve to the existing entity
        let alice_resolution = result
            .entities
            .iter()
            .find(|e| e.extracted.name.to_lowercase() == "alice");
        assert!(alice_resolution.is_some(), "alice entity should be present");
        let alice = alice_resolution.unwrap();
        assert_eq!(alice.resolution, ResolutionResult::Same);
        assert_eq!(alice.matched_id.as_deref(), Some("alice"));

        // The works_at fact should be detected as an Update (alice moves from acme_corp to newco)
        assert!(!result.facts.is_empty(), "pipeline should produce facts");
        let works_at = result
            .facts
            .iter()
            .find(|f| f.extracted.predicate == "works_at");
        assert!(works_at.is_some(), "works_at fact should be present");
        let wf = works_at.unwrap();
        assert_eq!(wf.relation, FactRelation::Update);
        assert_eq!(wf.conflicting_fact_id, Some(1));
    }

    #[tokio::test]
    async fn test_pipeline_new_entity_not_resolved() {
        let pipeline =
            IntelligencePipeline::new(MockExtractor, MockResolver, MockContradictionDetector);

        let result = pipeline
            .process("Bob manages Alice", &[], &[])
            .await
            .unwrap();

        // No existing entities to resolve against — all should be Different
        for resolved in &result.entities {
            assert_eq!(
                resolved.resolution,
                ResolutionResult::Different,
                "entity '{}' should not resolve against empty set",
                resolved.extracted.name
            );
            assert!(resolved.matched_id.is_none());
        }

        // Fact should be Consistent with no existing facts to conflict with
        for fact in &result.facts {
            assert_eq!(fact.relation, FactRelation::Consistent);
            assert!(fact.conflicting_fact_id.is_none());
        }
    }
}
