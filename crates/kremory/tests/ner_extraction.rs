/// Integration tests for GlinerExtractor — GLiNER-based zero-shot NER.
///
/// All tests are `#[ignore]` because they require downloading
/// `onnx-community/gliner_large-v2.1` from HuggingFace Hub (~653 MB INT8 model).
///
/// The model is cached by hf-hub after the first download; subsequent runs use
/// the local cache at `~/.cache/huggingface/hub/`.
///
/// Run with:
/// ```
/// HF_HOME=~/.cache/huggingface cargo test --features ner --test ner_extraction -- --include-ignored --nocapture
/// ```
#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "ner")]
mod ner_tests {
    use kremory::core::{
        intelligence::{EntityExtractor, ExtractionContext},
        ner::GlinerExtractor,
    };
    use metrics_util::debugging::DebuggingRecorder;

    use super::common;

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(f)
    }

    /// The full the host application entity type list — shared across all NER tests.
    /// This is the same list set in `rust-pipeline/src/main.rs`.
    fn the-host-application_entity_types() -> Vec<String> {
        vec![
            "person".into(),
            "organisation".into(),
            "location".into(),
            "project".into(),
            "product".into(),
            "technology".into(),
            "role".into(),
            "event".into(),
            "date".into(),
            "money".into(),
            "document".into(),
            "metric".into(),
            "regulation".into(),
        ]
    }

    // ─── Ignored tests (require HF model download) ────────────────────────────

    /// Smoke test: load the model and extract entities from a meeting transcript snippet.
    ///
    /// Verifies that at least 3 entities are found, that "Alice Johnson" and
    /// "Anthropic" are among them, that all entities have non-empty names/labels,
    /// that confidence scores are in [0, 1], and that no facts are returned.
    #[test]
    #[ignore = "requires onnx-community/gliner_large-v2.1 download (~653 MB)"]
    fn test_gliner_meeting_transcript_extraction() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);

        let extractor = GlinerExtractor::new().expect("model load failed");

        let transcript =
            "Alice Johnson from Anthropic presented the Q4 roadmap for Project Atlas. \
            Bob Chen from the engineering team asked about the timeline for deploying to AWS. \
            The team agreed to meet again in San Francisco next Tuesday.";

        let meeting_types = the-host-application_entity_types();
        let ctx = ExtractionContext {
            allowed_entity_types: &meeting_types,
            ..ExtractionContext::default()
        };
        let start = std::time::Instant::now();
        let result = block_on(extractor.extract(transcript, &ctx)).expect("extraction failed");
        let elapsed = start.elapsed();

        eprintln!(
            "\n=== NER EXTRACTION RESULTS ({:.0}ms) ===",
            elapsed.as_secs_f64() * 1000.0
        );
        for e in &result.entities {
            eprintln!(
                "  [{:>12}] {} (conf: {:.2})",
                e.label,
                e.name,
                e.properties
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            );
        }
        eprintln!(
            "  Total: {} entities, {} facts",
            result.entities.len(),
            result.facts.len()
        );
        eprintln!("=========================================\n");

        assert!(
            result.entities.len() >= 3,
            "expected at least 3 entities, got {}: {:?}",
            result.entities.len(),
            result.entities.iter().map(|e| &e.name).collect::<Vec<_>>()
        );

        let names: Vec<&str> = result.entities.iter().map(|e| e.name.as_str()).collect();

        let found_alice = names
            .iter()
            .any(|n| n.contains("Alice") || n.contains("Johnson"));
        assert!(
            found_alice,
            "expected 'Alice Johnson' in entities, got: {names:?}"
        );

        let found_anthropic = names.iter().any(|n| n.contains("Anthropic"));
        assert!(
            found_anthropic,
            "expected 'Anthropic' in entities, got: {names:?}"
        );

        for entity in &result.entities {
            assert!(!entity.name.is_empty(), "entity name must not be empty");
            assert!(!entity.label.is_empty(), "entity label must not be empty");
            let confidence = entity.properties["confidence"]
                .as_f64()
                .expect("confidence must be a number");
            assert!(
                (0.0..=1.0).contains(&confidence),
                "confidence {confidence} out of [0, 1]"
            );
        }

        // NER extractor does not produce facts — relationship extraction is LLM-only.
        assert!(
            result.facts.is_empty(),
            "GlinerExtractor must return no facts, got: {:?}",
            result.facts
        );

        let snapshot = snapshotter.snapshot().into_vec();
        let extraction_ms = snapshot
            .iter()
            .find(|(k, _, _, _)| k.key().name() == "rql.ner.extraction_ms");
        assert!(
            extraction_ms.is_some(),
            "rql.ner.extraction_ms metric must be recorded"
        );

        let entity_count = snapshot
            .iter()
            .find(|(k, _, _, _)| k.key().name() == "rql.ner.entity_count");
        assert!(
            entity_count.is_some(),
            "rql.ner.entity_count metric must be recorded"
        );

        let exporter = common::MetricsExporter::new("logs");
        exporter
            .export(&snapshotter, "ner-meeting-transcript")
            .expect("metrics export failed");
    }

    /// Verify that `allowed_entity_types` in ExtractionContext restricts output labels.
    #[test]
    #[ignore = "requires onnx-community/gliner_large-v2.1 download (~653 MB)"]
    fn test_gliner_allowed_entity_types_override() {
        let extractor = GlinerExtractor::new().expect("model load failed");

        let transcript = "Alice from Acme Corp joined Project Falcon last Monday in Berlin.";

        let allowed = vec!["person".to_string()];
        let ctx = ExtractionContext {
            allowed_entity_types: &allowed,
            ..ExtractionContext::default()
        };

        let result = block_on(extractor.extract(transcript, &ctx)).expect("extraction failed");

        for entity in &result.entities {
            assert_eq!(
                entity.label, "person",
                "only 'person' labels expected when allowed_entity_types=[\"person\"], got: {}",
                entity.label
            );
        }
    }

    /// Verify that `excluded_entity_types` removes matching labels from output.
    #[test]
    #[ignore = "requires onnx-community/gliner_large-v2.1 download (~653 MB)"]
    fn test_gliner_excluded_entity_types() {
        let extractor = GlinerExtractor::new().expect("model load failed");

        let transcript = "Alice from Acme Corp joined Project Falcon.";

        let allowed = vec![
            "person".to_string(),
            "organisation".to_string(),
            "location".to_string(),
            "project".to_string(),
        ];
        let excluded = vec!["organisation".to_string()];
        let ctx = ExtractionContext {
            allowed_entity_types: &allowed,
            excluded_entity_types: &excluded,
            ..ExtractionContext::default()
        };

        let result = block_on(extractor.extract(transcript, &ctx)).expect("extraction failed");

        for entity in &result.entities {
            assert_ne!(
                entity.label, "organisation",
                "organisation label must be excluded, but got entity: {:?}",
                entity
            );
        }
    }

    /// When all effective entity types are excluded, the extractor returns empty
    /// without running model inference.
    #[test]
    #[ignore = "requires onnx-community/gliner_large-v2.1 download (~653 MB)"]
    fn test_gliner_all_types_excluded_returns_empty() {
        let extractor = GlinerExtractor::new().expect("model load failed");

        // Provide types then exclude all of them — effective list is empty.
        let allowed = vec![
            "person".to_string(),
            "organisation".to_string(),
            "location".to_string(),
        ];
        let excluded = vec![
            "person".to_string(),
            "organisation".to_string(),
            "location".to_string(),
        ];
        let ctx = ExtractionContext {
            allowed_entity_types: &allowed,
            excluded_entity_types: &excluded,
            ..ExtractionContext::default()
        };

        let result = block_on(extractor.extract("Alice joined Acme Corp.", &ctx))
            .expect("extraction failed");

        assert!(
            result.entities.is_empty(),
            "all types excluded → no entities expected"
        );
        assert!(
            result.facts.is_empty(),
            "all types excluded → no facts expected"
        );
    }

    /// Verify that GLiNER only returns labels from the provided entity type list,
    /// across diverse meeting scenarios (sales, legal, finance, hiring, medical).
    #[test]
    #[ignore = "requires onnx-community/gliner_large-v2.1 download (~653 MB)"]
    fn test_gliner_diverse_meetings_labels_within_allowed() {
        let extractor = GlinerExtractor::new().expect("model load failed");
        let types = the-host-application_entity_types();

        let scenarios: Vec<(&str, &str, Vec<&str>)> = vec![
            (
                "sales_call",
                "Sarah from Acme Corp discussed the $250K deal for Enterprise Platform. \
                 The VP of Sales wants to close by March 15th. Their competitor Salesforce \
                 was mentioned as the alternative.",
                vec![
                    "Sarah",
                    "Acme Corp",
                    "$250K",
                    "Enterprise Platform",
                    "Salesforce",
                ],
            ),
            (
                "legal_review",
                "Attorney James Park reviewed the NDA with CloudTech Inc. \
                 The GDPR compliance clause needs updating before the May deadline. \
                 The contract references SOC 2 Type II certification.",
                vec!["James Park", "CloudTech", "NDA", "GDPR"],
            ),
            (
                "finance_board",
                "CFO Maria Chen presented the Q3 earnings report showing ARR of $12M. \
                 The board discussed the Series B raise with Sequoia Capital in New York. \
                 Churn rate dropped to 2.1% from 3.4% last quarter.",
                vec!["Maria Chen", "Sequoia Capital", "New York"],
            ),
            (
                "hiring_panel",
                "Hiring Manager David Liu interviewed the candidate for Senior Engineer. \
                 They discussed experience with Kubernetes and React at their previous \
                 role at Google. Start date would be January 6th.",
                vec!["David Liu", "Google"],
            ),
            (
                "medical_consult",
                "Dr. Emily Watson from St. Mary's Hospital reviewed the patient outcomes \
                 report. The new HIPAA guidelines require updated consent forms. \
                 The clinical trial budget is $1.2M through December.",
                vec!["Emily Watson", "St. Mary's Hospital", "HIPAA"],
            ),
        ];

        for (scenario_name, transcript, expected_substrings) in &scenarios {
            let ctx = ExtractionContext {
                allowed_entity_types: &types,
                ..ExtractionContext::default()
            };
            let result = block_on(extractor.extract(transcript, &ctx))
                .unwrap_or_else(|e| panic!("{scenario_name}: extraction failed: {e}"));

            eprintln!(
                "\n=== {scenario_name} ({} entities) ===",
                result.entities.len()
            );
            for e in &result.entities {
                let conf = e
                    .properties
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                eprintln!("  [{:>13}] {} (conf: {:.2})", e.label, e.name, conf);
            }

            // Every returned label must be in our allowed list.
            for entity in &result.entities {
                assert!(
                    types.iter().any(|t| t == &entity.label),
                    "{scenario_name}: entity '{}' has label '{}' which is not in allowed types",
                    entity.name,
                    entity.label,
                );
            }

            // At least some of the expected entities should be found.
            let names: Vec<String> = result
                .entities
                .iter()
                .map(|e| e.name.to_lowercase())
                .collect();
            let mut found_count = 0;
            for expected in expected_substrings {
                if names.iter().any(|n| n.contains(&expected.to_lowercase())) {
                    found_count += 1;
                }
            }
            assert!(
                found_count >= expected_substrings.len() / 2,
                "{scenario_name}: expected at least half of {expected_substrings:?} to be found, \
                 but only {found_count}/{} matched. Found: {names:?}",
                expected_substrings.len(),
            );
        }
    }

    /// Metrics are recorded on every successful extraction, including short inputs.
    #[test]
    #[ignore = "requires onnx-community/gliner_large-v2.1 download (~653 MB)"]
    fn test_gliner_metrics_recorded_on_short_input() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);

        let extractor = GlinerExtractor::new().expect("model load failed");

        let meeting_types = vec!["person".to_string(), "organisation".to_string()];
        let ctx = ExtractionContext {
            allowed_entity_types: &meeting_types,
            ..ExtractionContext::default()
        };
        let _ = block_on(extractor.extract("Alice joined Acme.", &ctx)).expect("extraction failed");

        let snapshot = snapshotter.snapshot().into_vec();

        assert!(
            snapshot
                .iter()
                .any(|(k, _, _, _)| k.key().name() == "rql.ner.extraction_ms"),
            "rql.ner.extraction_ms must be recorded"
        );
        assert!(
            snapshot
                .iter()
                .any(|(k, _, _, _)| k.key().name() == "rql.ner.entity_count"),
            "rql.ner.entity_count must be recorded"
        );

        let exporter = common::MetricsExporter::new("logs");
        exporter
            .export(&snapshotter, "ner-short-input")
            .expect("metrics export failed");
    }
}
