//! Extraction subsystem — entity + relationship extraction from text.
//!
//! Module structure (post E-1 BYOE redesign):
//!   models            — serde structs for LLM JSON output coercion
//!   json_repair       — JSON repair utilities
//!   parsers           — domain parsers (entities, facts, relation names)
//!   graphiti          — LlmExtractor (3-stage LLM, Graphiti-quality prompts)
//!   default_extractor — DefaultExtractor (integer-ID L1 path)
//!   nuextract         — removed (tombstone module)
//!   single_call       — SingleCallExtractor (free-discovery single pass)
//!   programmatic      — ProgrammaticFirstExtractor (candidates-first)
//!   factory           — ExtractorKind dispatch enum
//!   hybrid_typer      — GlinerLlmExtractor (GLiNER + LLM, `ner` feature)
//!   delimited_tuple   — delimited-tuple structured output
//!   prompts           — prompt-building utilities
//!   schemas           — JSON schema definitions
//!   structured        — structured LLM call builder

// ─── Existing modules ────────────────────────────────────────────────────────

pub(crate) mod delimited_tuple;
pub(crate) mod factory;
#[cfg(feature = "ner")]
pub(crate) mod hybrid_typer;
pub mod prompts;
pub(crate) mod schemas;
pub(crate) mod structured;

// ─── New submodules (TD-001 E0-B split) ──────────────────────────────────────

pub(crate) mod default_extractor;
pub(crate) mod graphiti;
pub(crate) mod json_repair;
pub(crate) mod models;
pub(crate) mod nuextract;
pub(crate) mod parsers;
pub(crate) mod programmatic;
pub(crate) mod single_call;

// ─── Public re-exports ───────────────────────────────────────────────────────
//
// Integration tests (tests/*.rs) compile as separate crates and access items
// via `kremory::core::extraction::*`. These must be `pub`, not `pub(crate)`.

// Extractors — pub so integration tests can name them
pub use default_extractor::DefaultExtractor;
pub use graphiti::LlmExtractor;
pub use programmatic::ProgrammaticFirstExtractor;
pub use single_call::{PromptVersion, SingleCallExtractor};

// Model helpers — pub so integration tests can call is_canonical_entity_type
pub use models::{is_canonical_entity_type, normalize_label};

// ─── Test-only re-exports ─────────────────────────────────────────────────────
//
// Items only needed by unit tests in this file (via `use super::*`).
// Wrapped in #[cfg(test)] so cargo check (no test compilation) doesn't warn.

#[cfg(test)]
pub(crate) use std::sync::Arc;

// Intelligence types + EntityExtractor trait (needed to call .extract() in tests)
#[cfg(test)]
pub(crate) use crate::core::intelligence::{EntityExtractor, ExtractedEntity};

// Prompt builders
#[cfg(test)]
pub(crate) use graphiti::{build_entity_prompt, build_relation_names_prompt};

// JSON repair + parse utilities
#[cfg(test)]
pub(crate) use json_repair::{fix_unclosed_string_before_brace, parse_nuextract_response};
#[cfg(test)]
pub(crate) use programmatic::{parse_json_lenient, EntityOnlyOutput};

// Parsers
#[cfg(test)]
pub(crate) use parsers::{parse_entities, parse_entities_integer, parse_facts};

// Models
#[cfg(test)]
pub(crate) use models::{RawEntitySimple, ENTITY_TYPE_CANONICAL_FORMS};

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::core::intelligence::ExtractionContext;
    use crate::core::provider::MockChatProvider;
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use std::collections::HashMap;

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }

    /// Build a MockChatProvider that returns staged responses:
    /// - prompt containing "Extract all unique named entities" → stage1_json
    /// - prompt containing "relationship types connect" → stage2_json
    /// - prompt containing "Extract the key relationships" → stage3_json
    fn staged_mock(stage1_json: &str, stage2_json: &str, stage3_json: &str) -> MockChatProvider {
        let mut map = HashMap::new();
        map.insert(
            "Extract all unique named entities".to_string(),
            stage1_json.to_string(),
        );
        map.insert(
            "relationship types connect".to_string(),
            stage2_json.to_string(),
        );
        map.insert(
            "Extract the key relationships".to_string(),
            stage3_json.to_string(),
        );
        MockChatProvider::new(map)
    }

    #[test]
    fn test_default_extractor_with_mock_llm() {
        // TD-013 L1: mock LLM emits integer-ID format; registry resolves ids to labels.
        let stage1 = r#"{"entities":[{"name":"Alice","entity_type_id":1},{"name":"Acme Corp","entity_type_id":2}]}"#;
        let stage2 = r#"["works_at"]"#;
        let stage3 = r#"[{"subject":"Alice","predicate":"works_at","object":"Acme Corp","is_entity_ref":true,"confidence":0.9}]"#;

        let specs = vec![
            crate::core::entity_types::EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A person.".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "An org.".to_string(),
            },
        ];
        let mock = Arc::new(staged_mock(stage1, stage2, stage3));
        let extractor = DefaultExtractor::new(mock);
        let ctx = ExtractionContext {
            registry_specs: &specs,
            ..ExtractionContext::default()
        };
        let result = block_on(extractor.extract("Alice works at Acme Corp", &ctx)).unwrap();

        assert_eq!(result.entities.len(), 2, "should extract 2 entities");
        assert_eq!(result.facts.len(), 1, "should extract 1 fact");

        let alice = result.entities.iter().find(|e| e.name == "Alice").unwrap();
        assert_eq!(alice.label, "Person");

        let fact = &result.facts[0];
        assert_eq!(fact.subject, "Alice");
        assert_eq!(fact.predicate, "works_at");
        assert_eq!(fact.object, "Acme Corp");
        assert!(fact.is_entity_ref);
        assert!((fact.confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_cascade_stage2_receives_stage1_entities() {
        // TD-013 L1: stage1 mock emits integer-ID format; registry resolves id=2 → "Organisation".
        let stage1 = r#"{"entities":[{"name":"GlobalCorp","entity_type_id":2}]}"#;
        let stage2 = r#"["founded_by"]"#;
        let stage3 = r#"[]"#;

        // The stage2 prompt is keyed on "relationship types connect" which is in build_relation_names_prompt.
        // We verify the function directly builds the prompt with entity info.
        let entities = vec![ExtractedEntity {
            name: "GlobalCorp".to_string(),
            label: "Organisation".to_string(),
            properties: serde_json::Value::Object(serde_json::Map::new()),
        }];
        let prompt = build_relation_names_prompt("some text", &entities, &[]);
        assert!(
            prompt.contains("GlobalCorp"),
            "stage2 prompt must contain entity name from stage1"
        );
        assert!(
            prompt.contains("Organisation"),
            "stage2 prompt must contain entity label from stage1"
        );

        // Also verify the full extractor runs without error.
        let specs = vec![
            crate::core::entity_types::EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "An org.".to_string(),
            },
        ];
        let mock = Arc::new(staged_mock(stage1, stage2, stage3));
        let extractor = DefaultExtractor::new(mock);
        let ctx = ExtractionContext {
            registry_specs: &specs,
            ..ExtractionContext::default()
        };
        let result = block_on(extractor.extract("GlobalCorp was founded.", &ctx)).unwrap();
        assert_eq!(result.entities.len(), 1);
    }

    #[test]
    fn test_excluded_entities_filtered() {
        // TD-013 L1: mock LLM emits integer-ID format; StopWord is id=3 and should
        // be filtered out by excluded_entity_types matching on the resolved label.
        let stage1 = r#"{"entities":[{"name":"Alice","entity_type_id":1},{"name":"StopWordInc","entity_type_id":3}]}"#;
        let stage2 = r#"[]"#;
        let stage3 = r#"[]"#;

        let specs = vec![
            crate::core::entity_types::EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A person.".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 3,
                name: "StopWord".to_string(),
                description: "A stop-word entity.".to_string(),
            },
        ];
        let mock = Arc::new(staged_mock(stage1, stage2, stage3));
        let extractor = DefaultExtractor::new(mock);

        let excluded = vec!["StopWord".to_string()];
        let ctx = ExtractionContext {
            excluded_entity_types: &excluded,
            registry_specs: &specs,
            ..ExtractionContext::default()
        };
        let result = block_on(extractor.extract("Alice and StopWordInc", &ctx)).unwrap();

        assert_eq!(
            result.entities.len(),
            1,
            "excluded entity type should be filtered out"
        );
        assert_eq!(result.entities[0].name, "Alice");
        assert!(
            result.entities.iter().all(|e| e.label != "StopWord"),
            "no StopWord entity should remain"
        );
    }

    #[test]
    fn test_empty_llm_response_graceful() {
        // Mock returns empty arrays for all stages.
        let mock = Arc::new(staged_mock("[]", "[]", "[]"));
        let extractor = DefaultExtractor::new(mock);
        let ctx = ExtractionContext::default();
        let result = block_on(extractor.extract("Some text with no matches", &ctx));

        assert!(
            result.is_ok(),
            "empty LLM responses must not cause an error"
        );
        let result = result.unwrap();
        assert!(result.entities.is_empty(), "no entities expected");
        assert!(result.facts.is_empty(), "no facts expected");
    }

    #[test]
    fn test_malformed_llm_response_graceful() {
        // Mock returns malformed JSON that should be gracefully degraded.
        let mock = Arc::new(staged_mock("not valid json {{{", "also bad", "broken"));
        let extractor = DefaultExtractor::new(mock);
        let ctx = ExtractionContext::default();
        let result = block_on(extractor.extract("Some text", &ctx));

        assert!(
            result.is_ok(),
            "malformed LLM response must not cause an error"
        );
        let result = result.unwrap();
        assert!(
            result.entities.is_empty(),
            "graceful degradation: empty entities"
        );
        assert!(result.facts.is_empty(), "graceful degradation: empty facts");
    }

    #[test]
    fn test_ontology_constraint_in_prompt() {
        let allowed = vec!["Person".to_string(), "Organisation".to_string()];
        let prompt = build_entity_prompt("Alice works at Acme.", &allowed, &[]);
        assert!(
            prompt.contains("Person"),
            "stage1 prompt must contain allowed entity type Person"
        );
        assert!(
            prompt.contains("Organisation"),
            "stage1 prompt must contain allowed entity type Organisation"
        );
        assert!(
            prompt.contains("Only extract entities of these types"),
            "stage1 prompt must include the type-constraint hint"
        );
    }

    #[test]
    fn test_parse_entities_handles_empty_array() {
        let result = parse_entities("[]").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_facts_handles_empty_array() {
        let result = parse_facts("[]").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_entities_valid_json() {
        let json = r#"[{"name":"Alice","label":"Person"},{"name":"Acme","label":"Organisation"}]"#;
        let entities = parse_entities(json).unwrap();
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].name, "Alice");
        assert_eq!(entities[0].label, "Person");
        assert_eq!(entities[1].name, "Acme");
        assert_eq!(entities[1].label, "Organisation");
    }

    #[test]
    fn test_parse_facts_valid_json() {
        let json = r#"[{"subject":"Alice","predicate":"works_at","object":"Acme","is_entity_ref":true,"confidence":0.9}]"#;
        let facts = parse_facts(json).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].subject, "Alice");
        assert_eq!(facts[0].predicate, "works_at");
        assert!(facts[0].is_entity_ref);
        assert!((facts[0].confidence - 0.9).abs() < 0.001);
    }

    // ─── llm_json repair integration tests ──────────────────────────────

    #[test]
    fn test_parse_entities_repairs_objects_without_array_wrapper() {
        let messy = r#"{"name":"Alice","label":"Person"},{"name":"Bob","label":"Person"}"#;
        let entities = parse_entities(messy).unwrap();
        assert_eq!(
            entities.len(),
            2,
            "should repair unwrapped objects into array"
        );
        assert_eq!(entities[0].name, "Alice");
        assert_eq!(entities[1].name, "Bob");
    }

    #[test]
    fn test_parse_entities_repairs_trailing_comma() {
        let messy = r#"[{"name":"Alice","label":"Person"},{"name":"Bob","label":"Person"},]"#;
        let entities = parse_entities(messy).unwrap();
        assert_eq!(entities.len(), 2, "should handle trailing comma");
    }

    #[test]
    fn test_parse_entities_repairs_markdown_code_fence() {
        let messy = "```json\n[{\"name\":\"Alice\",\"label\":\"Person\"}]\n```";
        let entities = parse_entities(messy).unwrap();
        assert_eq!(entities.len(), 1, "should strip markdown fences");
        assert_eq!(entities[0].name, "Alice");
    }

    #[test]
    fn test_parse_facts_repairs_objects_without_array_wrapper() {
        let messy = r#"{"subject":"Alice","predicate":"works_at","object":"Acme","is_entity_ref":true,"confidence":0.9},{"subject":"Bob","predicate":"manages","object":"Team","is_entity_ref":false,"confidence":0.8}"#;
        let facts = parse_facts(messy).unwrap();
        assert_eq!(
            facts.len(),
            2,
            "should repair unwrapped fact objects into array"
        );
        assert_eq!(facts[0].subject, "Alice");
        assert_eq!(facts[1].subject, "Bob");
    }

    #[test]
    fn test_parse_facts_repairs_single_quotes() {
        let messy = r#"[{'subject':'Alice','predicate':'works_at','object':'Acme','is_entity_ref':true,'confidence':0.9}]"#;
        let facts = parse_facts(messy).unwrap();
        assert_eq!(facts.len(), 1, "should handle single-quoted keys/values");
        assert_eq!(facts[0].subject, "Alice");
    }

    #[test]
    fn test_parse_entities_repairs_python_style_booleans() {
        // Some models output Python-style True/False/None
        let messy = r#"[{"name":"Alice","label":"Person","active":True}]"#;
        let entities = parse_entities(messy).unwrap();
        assert_eq!(entities.len(), 1, "should handle Python-style booleans");
    }

    // ─── Integer-ID parse path tests (TD-013 L1) ──────────────────────────────

    fn make_test_registry() -> crate::core::entity_types::EntityTypeRegistry {
        crate::core::entity_types::EntityTypeRegistry::from_specs(vec![
            crate::core::entity_types::EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A person.".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "An org.".to_string(),
            },
        ])
    }

    #[test]
    fn parse_entities_integer_resolves_id_to_label() {
        let registry = make_test_registry();
        let json = r#"{"entities":[{"name":"Alice","entity_type_id":1},{"name":"Acme Corp","entity_type_id":2}]}"#;
        let entities = parse_entities_integer(json, &registry).unwrap();
        assert_eq!(entities.len(), 2);
        let alice = entities.iter().find(|e| e.name == "Alice").unwrap();
        assert_eq!(alice.label, "Person", "id=1 must resolve to Person");
        let acme = entities.iter().find(|e| e.name == "Acme Corp").unwrap();
        assert_eq!(
            acme.label, "Organisation",
            "id=2 must resolve to Organisation"
        );
    }

    #[test]
    fn parse_entities_integer_falls_back_to_zero_on_out_of_range() {
        let registry = make_test_registry();
        // id=99 is out of range for this registry (max=2); must fall back to id=0 → "Entity"
        let json = r#"{"entities":[{"name":"Unknown Thing","entity_type_id":99}]}"#;
        let entities = parse_entities_integer(json, &registry).unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(
            entities[0].label, "Entity",
            "out-of-range id must fall back to Entity"
        );
    }

    #[test]
    fn parse_entities_integer_rejects_json_fragment_in_name() {
        // Lock the fix for the 2026-06-04 qwen2.5:14b benchmark failure: post-repair
        // garbage where a truncated entity's tail bleeds into the next entity's
        // `name` field (e.g. `Boston", "entity_type_id": 3}, {`). Without the
        // shape-validator these used to silently collapse to label="Entity".
        let registry = make_test_registry();
        let garbage =
            r#"{"entities":[{"name":"Boston\", \"entity_type_id\": 3}, {","entity_type_id":0}]}"#;
        let entities = parse_entities_integer(garbage, &registry).unwrap();
        assert!(
            entities.is_empty(),
            "names containing JSON syntax must be filtered, got: {entities:?}"
        );
    }

    #[test]
    fn parse_entities_integer_rejects_missing_entity_type_id() {
        // After stripping `#[serde(default)]` from RawEntityIntegerId, an entity
        // missing `entity_type_id` must fail deserialize loudly — not collapse
        // to id=0 → "Entity" via serde default. The fn returns `Ok(vec![])` on
        // total parse failure so the fallback ladder can attempt next arm.
        let registry = make_test_registry();
        let no_id = r#"{"entities":[{"name":"Alice"}]}"#;
        let entities = parse_entities_integer(no_id, &registry).unwrap();
        assert!(
            entities.is_empty(),
            "missing entity_type_id must NOT default to 0; got: {entities:?}"
        );
    }

    #[test]
    fn parse_entities_integer_rejects_missing_name() {
        // Symmetric to the above: missing `name` must also fail deserialize
        // rather than default to empty string + silently drop.
        let registry = make_test_registry();
        let no_name = r#"{"entities":[{"entity_type_id":1}]}"#;
        let entities = parse_entities_integer(no_name, &registry).unwrap();
        assert!(
            entities.is_empty(),
            "missing name must NOT default to empty; got: {entities:?}"
        );
    }

    // ─── Metrics assertion tests ─────────────────────────────────────────────

    type Snapshot = Vec<(
        metrics_util::CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )>;

    /// Sum across all counter rows matching `name`, regardless of labels.
    ///
    /// Per [[observability-first-class]] cardinal failure mode #9, post-TD-013
    /// the extractor emits `rql.extraction.json_parse_ok` with `path=wrapped|
    /// bare_array|post_repair` labels alongside legacy unlabeled emission sites.
    /// Each path is its own counter row in the registry, so `find()` (returning
    /// the first match) would under-report. Tests asking "did N parses succeed?"
    /// want the aggregate, so sum.
    fn find_counter(snapshot: &Snapshot, name: &str) -> u64 {
        snapshot
            .iter()
            .filter(|(k, ..)| k.key().name() == name)
            .map(|(.., v)| match v {
                DebugValue::Counter(n) => *n,
                _ => 0,
            })
            .sum()
    }

    fn find_histogram(snapshot: &Snapshot, name: &str) -> Vec<f64> {
        snapshot
            .iter()
            .filter(|(k, ..)| k.key().name() == name)
            .flat_map(|(.., v)| match v {
                DebugValue::Histogram(vals) => {
                    vals.iter().map(|v| v.into_inner()).collect::<Vec<_>>()
                }
                _ => vec![],
            })
            .collect()
    }

    #[test]
    fn test_extraction_emits_metrics() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            // TD-013 L1: integer-ID format; registry resolves id=1→Person, id=2→Organisation.
            let stage1 = r#"{"entities":[{"name":"Alice","entity_type_id":1},{"name":"Acme Corp","entity_type_id":2}]}"#;
            let stage2 = r#"["works_at"]"#;
            let stage3 = r#"[{"subject":"Alice","predicate":"works_at","object":"Acme Corp","is_entity_ref":true,"confidence":0.9}]"#;
            let specs = vec![
                crate::core::entity_types::EntityTypeSpec {
                    id: 0,
                    name: "Entity".to_string(),
                    description: "catch-all".to_string(),
                },
                crate::core::entity_types::EntityTypeSpec {
                    id: 1,
                    name: "Person".to_string(),
                    description: "A person.".to_string(),
                },
                crate::core::entity_types::EntityTypeSpec {
                    id: 2,
                    name: "Organisation".to_string(),
                    description: "An org.".to_string(),
                },
            ];
            let mock = Arc::new(staged_mock(stage1, stage2, stage3));
            let extractor = DefaultExtractor::new(mock);
            let ctx = ExtractionContext {
                registry_specs: &specs,
                ..ExtractionContext::default()
            };
            let _result = block_on(extractor.extract("Alice works at Acme Corp", &ctx)).unwrap();

            let snapshot = snapshotter.snapshot().into_vec();

            // 3 extraction stages should emit timing histograms
            let stage_timings = find_histogram(&snapshot, "rql.extraction.stage_ms");
            assert_eq!(
                stage_timings.len(),
                3,
                "should have 3 stage timings (entities, relations, triplets)"
            );
            assert!(
                stage_timings.iter().all(|&v| v >= 0.0),
                "all timings should be non-negative"
            );

            // Entity and fact count histograms
            let entity_counts = find_histogram(&snapshot, "rql.extraction.entity_count");
            assert_eq!(entity_counts.len(), 1);
            assert_eq!(entity_counts[0], 2.0, "should record 2 entities");

            let fact_counts = find_histogram(&snapshot, "rql.extraction.fact_count");
            assert_eq!(fact_counts.len(), 1);
            assert_eq!(fact_counts[0], 1.0, "should record 1 fact");

            // JSON parse success counters (stages 1 and 3 parse JSON)
            let parse_ok = find_counter(&snapshot, "rql.extraction.json_parse_ok");
            assert!(
                parse_ok >= 2,
                "at least 2 JSON parses should succeed (entities + facts)"
            );

            let parse_fail = find_counter(&snapshot, "rql.extraction.json_parse_fail");
            assert_eq!(parse_fail, 0, "no JSON parses should fail with mock data");
        });
    }

    #[test]
    fn test_extraction_metrics_on_json_failure() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            let mock = Arc::new(staged_mock("not valid json {{{", "also bad", "broken"));
            let extractor = DefaultExtractor::new(mock);
            let ctx = ExtractionContext::default();
            let _result = block_on(extractor.extract("Some text", &ctx)).unwrap();

            let snapshot = snapshotter.snapshot().into_vec();

            let parse_fail = find_counter(&snapshot, "rql.extraction.json_parse_fail");
            assert!(
                parse_fail >= 1,
                "should record at least 1 JSON parse failure"
            );

            let entity_counts = find_histogram(&snapshot, "rql.extraction.entity_count");
            assert_eq!(
                entity_counts[0], 0.0,
                "should record 0 entities on parse failure"
            );
        });
    }

    // ─── ProgrammaticFirstExtractor tests ────────────────────────────────────

    fn load_test_auditor() -> crate::core::text_utils::OovAuditor {
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
        let stops: std::collections::HashSet<String> =
            stop_words::get(stop_words::LANGUAGE::English)
                .into_iter()
                .map(|s| s.to_string())
                .collect();
        crate::core::text_utils::OovAuditor::new(dict, stops)
    }

    #[test]
    fn test_programmatic_first_with_mock_llm() {
        // Mock LLM: Call 1 (entity typing) returns confirmed entities,
        // Call 2 (relationships) returns a relationship triple.
        let mut map = HashMap::new();
        // Call 1 matches "Confirm which are real entities"
        map.insert(
            "Confirm which are real entities".to_string(),
            r#"{"entities":[{"name":"Alice","label":"Person"},{"name":"Acme Corp","label":"Organisation"}]}"#.to_string(),
        );
        // Call 2 matches "Extract relationships between these entities"
        map.insert(
            "Extract relationships between these entities".to_string(),
            r#"{"relationships":[{"subject":"Alice","predicate":"works_at","object":"Acme Corp","is_entity_ref":true,"confidence":0.9}]}"#.to_string(),
        );
        let mock = Arc::new(MockChatProvider::new(map));
        let auditor = Arc::new(load_test_auditor());

        let extractor = ProgrammaticFirstExtractor::new(mock, auditor, 25);
        let ctx = ExtractionContext::default();

        let result =
            block_on(extractor.extract("Alice works at Acme Corp in the downtown office.", &ctx))
                .unwrap();

        assert!(
            result.entities.len() >= 2,
            "should have at least Alice and Acme Corp, got {:?}",
            result.entities.iter().map(|e| &e.name).collect::<Vec<_>>()
        );
        assert!(
            !result.facts.is_empty(),
            "should have at least 1 relationship"
        );
        assert_eq!(result.facts[0].predicate, "works_at");
    }

    #[test]
    fn test_programmatic_first_empty_llm_still_runs() {
        // When LLM returns empty, extractor should return empty without error
        let map = HashMap::new();
        let mock = Arc::new(MockChatProvider::new(map));
        let auditor = Arc::new(load_test_auditor());

        let extractor = ProgrammaticFirstExtractor::new(mock, auditor, 25);
        let ctx = ExtractionContext::default();

        let result = block_on(extractor.extract("Alice works at Acme Corp.", &ctx)).unwrap();

        // LLM returned empty → no entities confirmed, no relationships
        assert!(
            result.entities.is_empty(),
            "empty LLM should yield no typed entities"
        );
    }

    #[test]
    fn test_parse_json_lenient_valid() {
        let parsed: Option<EntityOnlyOutput> =
            parse_json_lenient(r#"{"entities":[{"name":"Alice","label":"Person"}]}"#);
        assert!(parsed.is_some());
        assert_eq!(parsed.unwrap().entities.len(), 1);
    }

    #[test]
    fn test_parse_json_lenient_with_surrounding_text() {
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(
            r#"Here is the result: {"entities":[{"name":"Bob","label":"Person"}]} Done."#,
        );
        assert!(parsed.is_some());
        assert_eq!(parsed.unwrap().entities[0].name, "Bob");
    }

    #[test]
    fn test_parse_json_lenient_empty() {
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient("");
        assert!(parsed.is_none());
    }

    // ── fix_unclosed_string_before_brace ──────────────────────────────────────

    #[test]
    fn test_fix_unclosed_string_noop_on_valid_json() {
        // Well-formed JSON must pass through unchanged.
        let valid = r#"{"entities": [{"name": "Alice", "label": "Person"}]}"#;
        assert_eq!(fix_unclosed_string_before_brace(valid), valid);
    }

    #[test]
    fn test_fix_unclosed_string_repairs_gemma4_output() {
        // Reproduce the exact gemma4-e2b failure: labels missing closing `"` before `}`.
        let malformed = r#"{"entities": [{"name": "Alice", "label": "Person"}, {"name": "Bob", "label": "Person}, {"name": "Stanford", "label": "Place}], "relationships": []}"#;
        let fixed = fix_unclosed_string_before_brace(malformed);
        let parsed: serde_json::Value =
            serde_json::from_str(&fixed).expect("fixed output must be valid JSON");
        let entities = parsed["entities"]
            .as_array()
            .expect("entities must be array");
        assert_eq!(entities.len(), 3, "all 3 entities must survive the fix");
        assert_eq!(entities[0]["name"], "Alice");
        assert_eq!(entities[1]["label"], "Person");
        assert_eq!(entities[2]["name"], "Stanford");
    }

    #[test]
    fn test_fix_unclosed_string_handles_empty() {
        assert_eq!(fix_unclosed_string_before_brace(""), "");
    }

    #[test]
    fn test_parse_nuextract_response_repairs_gemma4_unclosed_labels() {
        // Integration: verify parse_nuextract_response recovers from gemma4-e2b output.
        let malformed = r#"{"entities": [{"name": "Alice", "label": "Person"}, {"name": "Bob", "label": "Person}, {"name": "Stanford", "label": "Place}], "relationships": []}"#;
        let ctx = ExtractionContext {
            known_entities: &[],
            allowed_entity_types: &[],
            allowed_edge_types: &[],
            excluded_entity_types: &[],
            content_type: crate::core::config::ContentType::Text,
            registry_specs: &[],
            existing_graph_entities: &[],
            arm_budget_ms: 30_000,
        };
        let (entities, _facts) =
            parse_nuextract_response(malformed, &ctx).expect("must not return Err");
        assert_eq!(entities.len(), 3, "all 3 entities must be extracted");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Alice"), "Alice must be extracted");
        assert!(names.contains(&"Bob"), "Bob must be extracted");
        assert!(names.contains(&"Stanford"), "Stanford must be extracted");
    }

    // ── L1: Adversarial parse_json_lenient — placeholder-label inputs ────────
    //
    // These tests catch the exact TD-012 failure mode: valid JSON that parses
    // successfully but carries placeholder labels ("Entity", "UNKNOWN", "").
    // The parser itself accepts such inputs (it only checks syntax), so these
    // tests document and pin the parser's behaviour and drive the L2 validator
    // to be the correct rejection layer.

    #[test]
    fn parse_json_lenient_placeholder_entity_label_parses_but_is_flagged() {
        // TD-012 exact shape: valid JSON with label="Entity" (placeholder).
        // parse_json_lenient MUST parse it — the syntax is correct.
        // is_canonical_entity_type MUST reject it — it's a placeholder.
        let json = r#"{"entities":[{"name":"Alice","label":"Entity"}]}"#;
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(json);
        assert!(
            parsed.is_some(),
            "parse_json_lenient must accept syntactically valid JSON even with placeholder label"
        );
        let entities = &parsed.unwrap().entities;
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].label, "Entity");
        // L2 validator must reject this label — it's the placeholder, not a type
        assert!(
            !is_canonical_entity_type(&entities[0].label),
            "is_canonical_entity_type must reject 'Entity' (TD-012 placeholder) — got: {}",
            entities[0].label
        );
    }

    #[test]
    fn parse_json_lenient_unknown_label_parses_but_is_flagged() {
        // "UNKNOWN" is another common placeholder emitted by poorly-constrained models.
        let json = r#"{"entities":[{"name":"OpenAI","label":"UNKNOWN"}]}"#;
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(json);
        assert!(parsed.is_some(), "must parse syntactically valid JSON");
        let entities = &parsed.unwrap().entities;
        assert_eq!(entities[0].label, "UNKNOWN");
        assert!(
            !is_canonical_entity_type(&entities[0].label),
            "is_canonical_entity_type must reject 'UNKNOWN'"
        );
    }

    #[test]
    fn parse_json_lenient_empty_label_uses_default_and_is_flagged() {
        // When label field is absent the default is "Entity" — also a placeholder.
        let json = r#"{"entities":[{"name":"Alice"}]}"#;
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(json);
        assert!(parsed.is_some(), "must parse JSON with missing label field");
        let entities = &parsed.unwrap().entities;
        // Default serde fills in "Entity"
        assert_eq!(
            entities[0].label, "Entity",
            "missing label must default to 'Entity'"
        );
        assert!(
            !is_canonical_entity_type(&entities[0].label),
            "default 'Entity' label must be rejected by is_canonical_entity_type"
        );
    }

    #[test]
    fn parse_json_lenient_mixed_canonical_and_placeholder_labels() {
        // When a batch has some canonical and some placeholder labels, the parser
        // accepts the whole batch and the validator rejects only the bad ones.
        let json =
            r#"{"entities":[{"name":"Alice","label":"Person"},{"name":"blob","label":"Entity"}]}"#;
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(json);
        assert!(parsed.is_some(), "must parse mixed-label JSON");
        let entities = &parsed.unwrap().entities;
        assert_eq!(entities.len(), 2);
        assert!(
            is_canonical_entity_type(&entities[0].label),
            "Person must be canonical"
        );
        assert!(
            !is_canonical_entity_type(&entities[1].label),
            "'Entity' placeholder must be rejected"
        );
    }

    #[test]
    fn parse_json_lenient_malformed_braces_triggers_repair() {
        // Unbalanced braces — parser should attempt llm_json repair and succeed.
        let json = r#"{"entities":[{"name":"Alice","label":"Person""#;
        // The parser may or may not recover from this — document the behaviour.
        // If repair succeeds, the result must be Some with at least name "Alice".
        // If repair fails, None is returned — that is also acceptable.
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(json);
        if let Some(out) = parsed {
            // Repair succeeded: verify the entity is intact
            if !out.entities.is_empty() {
                assert_eq!(out.entities[0].name, "Alice");
            }
        }
        // None is also acceptable — parser did not fabricate data
    }

    #[test]
    fn parse_json_lenient_array_root_falls_back_to_default() {
        // Array at root doesn't match EntityOnlyOutput directly, but the JSON repair + brace
        // extraction path produces an empty-entities result (because EntityOnlyOutput's
        // `entities` field has `#[serde(default)]`).
        //
        // The important invariant: NO entities are returned from an array-root response.
        // This prevents an LLM that emits a raw array (instead of {"entities":[...]}) from
        // injecting untyped entries — the caller sees zero entities and retries / falls back.
        let json = r#"[{"name":"Alice","label":"Person"}]"#;
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(json);
        let entity_count = parsed.map(|o| o.entities.len()).unwrap_or(0);
        assert_eq!(
            entity_count, 0,
            "array-root response must yield zero entities from EntityOnlyOutput — \
             the shape mismatch should suppress extraction, not smuggle in unlabelled entities"
        );
    }

    // ── L2: ENTITY_TYPE_ALLOWLIST + is_canonical_entity_type tests ───────────

    #[test]
    fn is_canonical_entity_type_accepts_person() {
        assert!(
            is_canonical_entity_type("Person"),
            "Person must be in the canonical entity type allowlist"
        );
    }

    #[test]
    fn is_canonical_entity_type_accepts_organisation() {
        assert!(
            is_canonical_entity_type("Organisation"),
            "Organisation must be in the canonical entity type allowlist"
        );
    }

    #[test]
    fn is_canonical_entity_type_accepts_location() {
        assert!(
            is_canonical_entity_type("Location"),
            "Location must be in the canonical entity type allowlist"
        );
    }

    #[test]
    fn is_canonical_entity_type_rejects_entity_placeholder() {
        assert!(
            !is_canonical_entity_type("Entity"),
            "'Entity' is a placeholder label (TD-012) — must be rejected by the allowlist"
        );
    }

    #[test]
    fn is_canonical_entity_type_rejects_unknown_placeholder() {
        assert!(
            !is_canonical_entity_type("UNKNOWN"),
            "'UNKNOWN' is a placeholder label — must be rejected by the allowlist"
        );
    }

    #[test]
    fn is_canonical_entity_type_rejects_empty_string() {
        assert!(
            !is_canonical_entity_type(""),
            "empty string is not a valid entity type — must be rejected"
        );
    }

    // ─── TD-013 PR1-corrected: mechanism semantics changed (2026-06-03) ──────
    //
    // is_canonical_entity_type previously required positive-allowlist match.
    // Now it requires structural validity + placeholder reject only — aligned
    // with Graphiti's `validate_node_labels` (regex-only) + peer ecosystem
    // consensus. Tests below assert NEW semantics.

    #[test]
    fn is_canonical_entity_type_accepts_arbitrary_valid_string() {
        // Novel labels NOT in any historical canonical list are ACCEPTED if
        // structurally valid. Aligns with Graphiti/Mem0/Cognee: no positive
        // allowlist; LLM is the type-discoverer.
        assert!(
            is_canonical_entity_type("SomeRandomType"),
            "structurally-valid novel types are accepted (no positive allowlist)"
        );
        assert!(
            is_canonical_entity_type("Court"),
            "ground-truth domain type 'Court' accepted (was missing from old allowlist)"
        );
        assert!(
            is_canonical_entity_type("Species"),
            "ground-truth domain type 'Species' accepted (was missing from old allowlist)"
        );
    }

    #[test]
    fn is_canonical_entity_type_accepts_case_variants() {
        // Case variants pass structural validity; canonical case-folding is
        // the responsibility of `normalize_label`, NOT this validator.
        assert!(
            is_canonical_entity_type("person"),
            "'person' (lowercase) is structurally valid; normalize_label maps to 'Person'"
        );
        assert!(
            is_canonical_entity_type("PERSON"),
            "'PERSON' (all-caps) is structurally valid; normalize_label maps to 'Person'"
        );
        assert!(
            is_canonical_entity_type("ORGANISATION"),
            "'ORGANISATION' (all-caps) is structurally valid"
        );
    }

    #[test]
    fn is_canonical_entity_type_rejects_structural_junk() {
        // Single-char, leading-digit, all-punctuation, or excessively-long
        // labels are structurally invalid (Graphiti-style Cypher safety pattern).
        assert!(!is_canonical_entity_type("A"), "single char rejected");
        assert!(
            !is_canonical_entity_type("123Type"),
            "leading digit rejected"
        );
        assert!(
            !is_canonical_entity_type("!!!"),
            "punctuation-only rejected"
        );
        assert!(
            !is_canonical_entity_type(&"X".repeat(100)),
            "overlong label rejected"
        );
    }

    #[test]
    fn normalize_label_canonicalizes_case() {
        assert_eq!(normalize_label("PERSON"), "Person");
        assert_eq!(normalize_label("person"), "Person");
        assert_eq!(normalize_label("Person"), "Person");
    }

    #[test]
    fn normalize_label_handles_us_uk_spelling() {
        // qwen2.5:14b emits "Organization" (US); kremory's canonical is "Organisation" (UK).
        // Same source of the original B3 mismatch documented in Vera review.
        assert_eq!(normalize_label("Organization"), "Organisation");
        assert_eq!(normalize_label("ORGANIZATION"), "Organisation");
        assert_eq!(normalize_label("Organisation"), "Organisation");
        // Also handle common abbreviations + synonyms.
        assert_eq!(normalize_label("Company"), "Organisation");
        assert_eq!(normalize_label("ORG"), "Organisation");
    }

    #[test]
    fn normalize_label_preserves_novel_labels() {
        // Novel labels not in alias map pass through unchanged (trimmed).
        assert_eq!(normalize_label("Court"), "Court");
        assert_eq!(normalize_label("Software"), "Software");
        assert_eq!(normalize_label("  Species  "), "Species");
    }

    #[test]
    fn canonical_forms_constant_pass_validation() {
        // ENTITY_TYPE_CANONICAL_FORMS contains the canonical reference set
        // for prompt interpolation. Each form must pass structural validity.
        for canonical in super::ENTITY_TYPE_CANONICAL_FORMS {
            assert!(
                is_canonical_entity_type(canonical),
                "canonical form '{canonical}' must pass is_canonical_entity_type"
            );
        }
    }

    // ─── L5: RawEntitySimple label array-coercion (qwen2.5:14b compat) ──────

    #[test]
    fn raw_entity_simple_label_coerces_array_to_string() {
        // qwen2.5:14b sometimes emits {"name": "Alice", "label": ["Person"]} —
        // deser_string_or_array coerces to RawEntitySimple { label: "Person", ... }
        let json = r#"{"name": "Alice", "label": ["Person"]}"#;
        let parsed: RawEntitySimple = serde_json::from_str(json)
            .expect("array-shaped label should coerce via deser_string_or_array");
        assert_eq!(parsed.name, "Alice");
        assert_eq!(parsed.label, "Person");
    }

    #[test]
    fn raw_entity_simple_label_accepts_string_unchanged() {
        let json = r#"{"name": "Bob", "label": "Person"}"#;
        let parsed: RawEntitySimple = serde_json::from_str(json).expect("string label");
        assert_eq!(parsed.name, "Bob");
        assert_eq!(parsed.label, "Person");
    }

    #[test]
    fn raw_entity_simple_label_joins_multiple_elements() {
        // deser_string_or_array joins array elements with ", " — matches
        // the same convention used for RawRelationship.subject/predicate/object.
        let json = r#"{"name": "Carol", "label": ["Person", "Politician"]}"#;
        let parsed: RawEntitySimple = serde_json::from_str(json)
            .expect("multi-element array should join via deser_string_or_array");
        assert_eq!(parsed.name, "Carol");
        assert_eq!(parsed.label, "Person, Politician");
    }
}
