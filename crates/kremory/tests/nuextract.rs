/// Unit tests for NuExtractExtractor — template-based single-pass extraction.
///
/// Tests the public API through EntityExtractor::extract() with MockChatProvider
/// returning NuExtract-style filled templates.
use std::collections::HashMap;
use std::sync::Arc;

use metrics_util::debugging::{DebugValue, DebuggingRecorder};
use kremory::core::config::ContentType;
use kremory::core::extraction::{GroundedNuExtractExtractor, NuExtractExtractor};
use kremory::core::intelligence::{EntityExtractor, ExtractionContext};
use kremory::core::provider::MockChatProvider;

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(f)
}

/// Build a MockChatProvider that returns `response` when the prompt contains "# Template:".
fn nuextract_mock(response: &str) -> MockChatProvider {
    let mut map = HashMap::new();
    map.insert("# Template:".to_string(), response.to_string());
    MockChatProvider::new(map)
}

// ─── Core extraction tests ───────────────────────────────────────────────────

#[test]
fn test_nuextract_basic_extraction() {
    let response = r#"{"entities": [{"name": "Alice", "label": "Person"}, {"name": "Acme Corp", "label": "Organisation"}], "relationships": [{"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"}]}"#;

    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default();
    let result = block_on(extractor.extract("Alice works at Acme Corp.", &ctx)).unwrap();

    assert_eq!(result.entities.len(), 2, "should extract 2 entities");
    assert_eq!(result.facts.len(), 1, "should extract 1 fact");

    let alice = result.entities.iter().find(|e| e.name == "Alice").unwrap();
    assert_eq!(alice.label, "Person");

    let acme = result
        .entities
        .iter()
        .find(|e| e.name == "Acme Corp")
        .unwrap();
    assert_eq!(acme.label, "Organisation");

    let fact = &result.facts[0];
    assert_eq!(fact.subject, "Alice");
    assert_eq!(fact.predicate, "works_at");
    assert_eq!(fact.object, "Acme Corp");
    assert!(
        fact.is_entity_ref,
        "object matches entity name → is_entity_ref=true"
    );
    assert!(
        (fact.confidence - 1.0).abs() < 0.001,
        "NuExtract verbatim → confidence=1.0"
    );
}

#[test]
fn test_nuextract_multiple_relationships() {
    let response = r#"{"entities": [{"name": "Alice", "label": "Person"}, {"name": "Bob", "label": "Person"}, {"name": "Project Phoenix", "label": "Project"}], "relationships": [{"subject": "Alice", "predicate": "manages", "object": "Project Phoenix"}, {"subject": "Bob", "predicate": "works_on", "object": "Project Phoenix"}]}"#;

    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default();
    let result = block_on(extractor.extract(
        "Alice manages Project Phoenix. Bob works on Project Phoenix.",
        &ctx,
    ))
    .unwrap();

    assert_eq!(result.entities.len(), 3);
    assert_eq!(result.facts.len(), 2);
    assert!(
        result.facts[0].is_entity_ref,
        "Project Phoenix is an entity"
    );
    assert!(
        result.facts[1].is_entity_ref,
        "Project Phoenix is an entity"
    );
}

#[test]
fn test_nuextract_is_entity_ref_false_for_non_entity_objects() {
    let response = r#"{"entities": [{"name": "Alice", "label": "Person"}], "relationships": [{"subject": "Alice", "predicate": "has_age", "object": "30"}]}"#;

    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default();
    let result = block_on(extractor.extract("Alice is 30 years old.", &ctx)).unwrap();

    assert_eq!(result.facts.len(), 1);
    assert!(
        !result.facts[0].is_entity_ref,
        "\"30\" is not an entity name"
    );
}

// ─── Organisation entity recovery ───────────────────────────────────────────

/// Regression test: NuExtract must recognise Organisation entities.
///
/// Before the enum label fix, NuExtract used "verbatim-string" for the label
/// field which caused the model to default to "Person" for all entities and
/// skip organisations. The fix adds an enum to the default template so the
/// model is guided to pick the correct type.
#[test]
fn test_nuextract_extracts_organisation_entities() {
    let response = r#"{"entities": [{"name": "Alice", "label": "Person"}, {"name": "Acme Corporation", "label": "Organisation"}], "relationships": [{"subject": "Alice", "predicate": "works_at", "object": "Acme Corporation"}]}"#;

    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default();
    let result = block_on(extractor.extract("Alice works at Acme Corporation.", &ctx)).unwrap();

    // Must find at least one Organisation entity.
    let org = result
        .entities
        .iter()
        .find(|e| e.label == "Organisation")
        .expect("should extract an Organisation entity");
    assert_eq!(org.name, "Acme Corporation");

    // The relationship object "Acme Corporation" matches the entity name, so
    // is_entity_ref must be true.
    let fact = result
        .facts
        .iter()
        .find(|f| f.object == "Acme Corporation")
        .expect("should have a fact whose object is Acme Corporation");
    assert!(
        fact.is_entity_ref,
        "Acme Corporation matches entity name → is_entity_ref=true"
    );
}

// ─── Empty / edge cases ─────────────────────────────────────────────────────

#[test]
fn test_nuextract_empty_response() {
    let response = r#"{"entities": [], "relationships": []}"#;

    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default();
    let result = block_on(extractor.extract("Nothing here.", &ctx)).unwrap();

    assert!(result.entities.is_empty());
    assert!(result.facts.is_empty());
}

#[test]
fn test_nuextract_entities_only_no_relationships() {
    let response = r#"{"entities": [{"name": "Alice", "label": "Person"}], "relationships": []}"#;

    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default();
    let result = block_on(extractor.extract("Alice was mentioned.", &ctx)).unwrap();

    assert_eq!(result.entities.len(), 1);
    assert!(result.facts.is_empty());
}

#[test]
fn test_nuextract_malformed_json_graceful() {
    let response = "not valid json at all";

    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default();
    let result = block_on(extractor.extract("Some text.", &ctx));

    assert!(result.is_ok(), "malformed JSON must degrade gracefully");
    let result = result.unwrap();
    assert!(result.entities.is_empty());
    assert!(result.facts.is_empty());
}

#[test]
fn test_nuextract_partial_json_missing_relationships_key() {
    let response = r#"{"entities": [{"name": "Alice", "label": "Person"}]}"#;

    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default();
    let result = block_on(extractor.extract("Alice mentioned.", &ctx)).unwrap();

    assert_eq!(result.entities.len(), 1, "entities should still parse");
    assert!(
        result.facts.is_empty(),
        "missing relationships key → no facts"
    );
}

#[test]
fn test_nuextract_skips_entities_with_empty_names() {
    let response = r#"{"entities": [{"name": "", "label": "Person"}, {"name": "Alice", "label": "Person"}], "relationships": []}"#;

    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default();
    let result = block_on(extractor.extract("Alice.", &ctx)).unwrap();

    assert_eq!(
        result.entities.len(),
        1,
        "empty-name entity should be skipped"
    );
    assert_eq!(result.entities[0].name, "Alice");
}

// ─── Exclusion filter ────────────────────────────────────────────────────────

#[test]
fn test_nuextract_excluded_entity_types() {
    let response = r#"{"entities": [{"name": "Alice", "label": "Person"}, {"name": "StopCorp", "label": "StopWord"}], "relationships": []}"#;

    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let excluded = vec!["StopWord".to_string()];
    let ctx = ExtractionContext {
        excluded_entity_types: &excluded,
        ..ExtractionContext::default()
    };
    let result = block_on(extractor.extract("Alice and StopCorp.", &ctx)).unwrap();

    assert_eq!(result.entities.len(), 1);
    assert_eq!(result.entities[0].name, "Alice");
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

type Snapshot = Vec<(
    metrics_util::CompositeKey,
    Option<metrics::Unit>,
    Option<metrics::SharedString>,
    DebugValue,
)>;

fn find_counter(snapshot: &Snapshot, name: &str) -> u64 {
    snapshot
        .iter()
        .find(|(k, ..)| k.key().name() == name)
        .map(|(.., v)| match v {
            DebugValue::Counter(n) => *n,
            _ => 0,
        })
        .unwrap_or(0)
}

fn find_histogram(snapshot: &Snapshot, name: &str) -> Vec<f64> {
    snapshot
        .iter()
        .filter(|(k, ..)| k.key().name() == name)
        .flat_map(|(.., v)| match v {
            DebugValue::Histogram(vals) => vals.iter().map(|v| v.into_inner()).collect::<Vec<_>>(),
            _ => vec![],
        })
        .collect()
}

#[test]
fn test_nuextract_emits_metrics() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    metrics::with_local_recorder(&recorder, || {
        let response = r#"{"entities": [{"name": "Alice", "label": "Person"}], "relationships": [{"subject": "Alice", "predicate": "works_at", "object": "Acme"}]}"#;
        let mock = Arc::new(nuextract_mock(response));
        let extractor = NuExtractExtractor::new(mock);
        let ctx = ExtractionContext::default();
        let _result = block_on(extractor.extract("Alice works at Acme.", &ctx)).unwrap();

        let snapshot = snapshotter.snapshot().into_vec();

        // Single-stage timing (not 3 like DefaultExtractor)
        let stage_timings = find_histogram(&snapshot, "rql.extraction.stage_ms");
        assert_eq!(
            stage_timings.len(),
            1,
            "NuExtract should emit exactly 1 stage timing"
        );

        let entity_counts = find_histogram(&snapshot, "rql.extraction.entity_count");
        assert_eq!(entity_counts[0], 1.0);

        let fact_counts = find_histogram(&snapshot, "rql.extraction.fact_count");
        assert_eq!(fact_counts[0], 1.0);

        let parse_ok = find_counter(&snapshot, "rql.extraction.json_parse_ok");
        assert_eq!(parse_ok, 1, "should record 1 successful JSON parse");

        let parse_fail = find_counter(&snapshot, "rql.extraction.json_parse_fail");
        assert_eq!(parse_fail, 0, "no parse failures expected");
    });
}

#[test]
fn test_nuextract_metrics_on_parse_failure() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    metrics::with_local_recorder(&recorder, || {
        let mock = Arc::new(nuextract_mock("NOT JSON AT ALL NO BRACES HERE"));
        let extractor = NuExtractExtractor::new(mock);
        let ctx = ExtractionContext::default();
        let _result = block_on(extractor.extract("Some text.", &ctx)).unwrap();
    });

    // Take snapshot AFTER exiting the metric recorder context
    let snapshot = snapshotter.snapshot().into_vec();

    let parse_fail = find_counter(&snapshot, "rql.extraction.json_parse_fail");
    assert!(parse_fail >= 1, "should record JSON parse failure");

    let entity_counts = find_histogram(&snapshot, "rql.extraction.entity_count");
    assert_eq!(entity_counts[0], 0.0, "0 entities on parse failure");
}

// ─── known_entities injection tests ──────────────────────────────────────────

#[test]
fn test_nuextract_known_entities_in_prompt() {
    // The mock key "Known entities:" only matches if known_entities is injected into the prompt.
    let mut map = HashMap::new();
    map.insert(
        "Known entities:".to_string(),
        r#"{"entities": [{"name": "Alice", "label": "Person"}], "relationships": []}"#.to_string(),
    );
    let mock = Arc::new(MockChatProvider::new(map));
    let extractor = NuExtractExtractor::new(mock);
    let known = vec![kremory::core::intelligence::ExtractedEntity {
        name: "Alice".into(),
        label: "Person".into(),
        properties: serde_json::json!({}),
    }];
    let ctx = ExtractionContext {
        known_entities: &known,
        ..ExtractionContext::default()
    };
    let result = block_on(extractor.extract("Alice spoke to Bob.", &ctx)).unwrap();

    assert_eq!(
        result.entities.len(),
        1,
        "known_entities hint should be present in prompt"
    );
    assert_eq!(result.entities[0].name, "Alice");
}

#[test]
fn test_nuextract_empty_known_entities_no_change() {
    // Empty known_entities → prompt must NOT contain "Known entities:" → backward compatible.
    // The mock only responds to "# Template:" (as the existing nuextract_mock does).
    let response = r#"{"entities": [{"name": "Alice", "label": "Person"}], "relationships": []}"#;
    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default(); // known_entities is &[]
    let result = block_on(extractor.extract("Alice.", &ctx)).unwrap();

    assert_eq!(
        result.entities.len(),
        1,
        "empty known_entities should not change extraction behaviour"
    );
    assert_eq!(result.entities[0].name, "Alice");
}

// ─── Content-type awareness tests ────────────────────────────────────────────

#[test]
fn test_extraction_context_carries_content_type() {
    let ctx_default = ExtractionContext::default();
    assert_eq!(
        ctx_default.content_type,
        ContentType::Text,
        "default content_type must be Text"
    );

    let ctx_message = ExtractionContext {
        content_type: ContentType::Message,
        ..ExtractionContext::default()
    };
    assert_eq!(ctx_message.content_type, ContentType::Message);

    let ctx_json = ExtractionContext {
        content_type: ContentType::Json,
        ..ExtractionContext::default()
    };
    assert_eq!(ctx_json.content_type, ContentType::Json);
}

// ─── GroundedNuExtractExtractor tests ────────────────────────────────────────

#[test]
fn test_grounded_extractor_two_pass() {
    let mut map = HashMap::new();
    // Pass 1 key: entity-only template contains `"entities": [{"name"`
    map.insert(
        r#""entities": [{"name""#.to_string(),
        r#"{"entities": [{"name": "Alice", "label": "Person"}, {"name": "Acme", "label": "Organisation"}]}"#.to_string(),
    );
    // Pass 2 key: relationship template contains `"relationships": [{"subject"`
    map.insert(
        r#""relationships": [{"subject""#.to_string(),
        r#"{"relationships": [{"subject": "Alice", "predicate": "works_at", "object": "Acme"}]}"#
            .to_string(),
    );
    let mock = Arc::new(MockChatProvider::new(map));
    let extractor = GroundedNuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default();
    let result = block_on(extractor.extract("Alice works at Acme.", &ctx)).unwrap();

    assert_eq!(result.entities.len(), 2, "should extract 2 entities");
    assert_eq!(result.facts.len(), 1, "should extract 1 fact");
    assert!(
        result.facts[0].is_entity_ref,
        "object 'Acme' is in entity set → is_entity_ref=true"
    );
}

#[test]
fn test_grounded_extractor_no_entities_skips_pass2() {
    let mut map = HashMap::new();
    // Pass 1 returns empty entities
    map.insert(
        r#""entities": [{"name""#.to_string(),
        r#"{"entities": []}"#.to_string(),
    );
    // Pass 2 should never be called; add a trap response that would produce facts if called
    map.insert(
        r#""relationships": [{"subject""#.to_string(),
        r#"{"relationships": [{"subject": "Alice", "predicate": "works_at", "object": "Acme"}]}"#
            .to_string(),
    );
    let mock = Arc::new(MockChatProvider::new(map));
    let extractor = GroundedNuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default();
    let result = block_on(extractor.extract("Nothing here.", &ctx)).unwrap();

    assert_eq!(result.entities.len(), 0, "no entities from Pass 1");
    assert_eq!(
        result.facts.len(),
        0,
        "Pass 2 must not be called when entities empty"
    );
}

#[test]
fn test_grounded_extractor_entity_constraint_in_template() {
    let mut map = HashMap::new();
    // Pass 1: return Alice and Acme
    map.insert(
        r#""entities": [{"name""#.to_string(),
        r#"{"entities": [{"name": "Alice", "label": "Person"}, {"name": "Acme", "label": "Organisation"}]}"#.to_string(),
    );
    // Pass 2 key: the template will contain entity names as enum values.
    // We match on "Alice" appearing in the prompt (only the constrained template has this).
    map.insert(
        "\"Alice\"".to_string(),
        r#"{"relationships": [{"subject": "Alice", "predicate": "works_at", "object": "Acme"}]}"#
            .to_string(),
    );
    let mock = Arc::new(MockChatProvider::new(map));
    let extractor = GroundedNuExtractExtractor::new(mock);
    let ctx = ExtractionContext::default();
    let result = block_on(extractor.extract("Alice works at Acme.", &ctx)).unwrap();

    // If Pass 2 used the entity-constrained template, "Alice" appears in the prompt
    // and the mock returns a fact — proving the constraint was present.
    assert_eq!(result.entities.len(), 2);
    assert_eq!(
        result.facts.len(),
        1,
        "entity names present in Pass 2 template → mock matched → 1 fact"
    );
}

#[test]
fn test_nuextract_message_type_extracts_successfully() {
    let response = r#"{"entities": [{"name": "Alice", "label": "Person"}, {"name": "Acme Corp", "label": "Organisation"}], "relationships": [{"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"}]}"#;

    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let ctx = ExtractionContext {
        content_type: ContentType::Message,
        ..ExtractionContext::default()
    };
    let result = block_on(extractor.extract("Speaker A: Alice works at Acme Corp.", &ctx)).unwrap();

    assert_eq!(
        result.entities.len(),
        2,
        "should extract 2 entities with Message content type"
    );
    assert_eq!(
        result.facts.len(),
        1,
        "should extract 1 fact with Message content type"
    );
}

#[test]
fn test_nuextract_json_type_extracts_successfully() {
    let response = r#"{"entities": [{"name": "Alice", "label": "Person"}], "relationships": []}"#;

    let mock = Arc::new(nuextract_mock(response));
    let extractor = NuExtractExtractor::new(mock);
    let ctx = ExtractionContext {
        content_type: ContentType::Json,
        ..ExtractionContext::default()
    };
    let result =
        block_on(extractor.extract(r#"{"user": "Alice", "role": "admin"}"#, &ctx)).unwrap();

    assert_eq!(
        result.entities.len(),
        1,
        "should extract 1 entity with Json content type"
    );
    assert!(result.facts.is_empty(), "no facts expected");
}
