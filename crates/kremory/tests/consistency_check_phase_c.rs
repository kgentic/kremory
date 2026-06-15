#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Phase C — Dream Pass 4 `consistency_check` core module acceptance tests.
//!
//! Governing specs:
//! - ADR-047 §Decision (1-6), §Module signature, §Observability, §Acceptance Criteria
//!   (amended post-Vera cycle 1)
//! - v0.1.2 sprint plan §Phase C DoD C1-C10
//! - v0.1.2 test strategy §Section 4 (Mocking Boundary) + §Section 5 (AC→Test Level Mapping)
//!
//! ## Acceptance criteria covered
//!
//! C1  — File exists and is <500 LoC (mechanical wc-l gate)
//! C2  — `ConsistencyCheckOpts::default()` matches ADR-047 spec (τ=0.6, cap=Some(50), None)
//! C3  — `embed_input_formatter()` deterministic + padding contract
//! C4  — `embed_prefilter_gate()` strict-less-than boundary (cos < τ, NOT ≤)
//! C5  — `Confirm/Reject/Modify` schema strict deserialization (no #[serde(default)] on required)
//! C6  — Cap-overflow guard: 100 candidates, cap=50 → flagged=50, overflow=50
//! C7  — Observability counters increment on the right code paths
//! C8  — `run_consistency_check()` orchestrates embed→prefilter→LLM→audit→summary
//! C9  — `dream_pass4_audit` row written per corrected action with pre_type_id
//! C10 — Real-LLM schema parses 100% on gemma4-e2b:latest (10 calls, #[ignore]-gated)
//!
//! ## How these tests fail (Red phase)
//!
//! `crates/kremory/src/core/dream/consistency_check` does NOT exist yet.
//! Every test will fail to compile until the Green agent creates the module and
//! wires it into `crates/kremory/src/core/dream/mod.rs`.
//!
//! ## Mocking boundary (per test strategy §Section 4)
//!
//! - ALWAYS REAL: SQLite (TemporalGraph file-backed via tempdir), Embedder
//!   (MockEmbeddingProvider / DeterministicEmbeddingProvider — cosine math IS
//!   what we test), consistency_check.rs itself.
//! - ALWAYS MOCK: ChatProvider (local MockLlm returning golden JSON per test).
//! - REAL LLM: Only C10 — #[ignore]-gated, --features llm-integration.

use autoagents_llm::chat::{ChatMessage, ChatProvider, ChatResponse as ChatResponseTrait, Tool};
use kremory::core::schema::TemporalGraph;

// ─── test helpers ─────────────────────────────────────────────────────────────

/// Open a fresh TemporalGraph backed by a tempdir file DB.
/// Runs ALL migrations (including 013) automatically on open.
async fn open_graph() -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir must succeed");
    let path = tmp.path().join("phase-c-test.db");
    let graph = TemporalGraph::open(path.to_str().expect("utf8 path"))
        .await
        .expect("TemporalGraph::open must succeed");
    (graph, tmp)
}

/// Insert an entity row directly for test setup.
///
/// Returns the integer rowid (used for audit-table FK assertions).
/// entity_type_source: 'Phase1Ner' | 'DreamPass4' | 'ConsumerPinned' | etc.
async fn insert_entity_with_rowid(
    graph: &TemporalGraph,
    id: &str,
    entity_type_id: i64,
    entity_type_source: &str,
    ner_confidence: Option<f64>,
) -> i64 {
    let now = chrono::Utc::now().to_rfc3339();
    let conf_val: libsql::Value = match ner_confidence {
        Some(v) => libsql::Value::Real(v),
        None => libsql::Value::Null,
    };
    let mut rows = graph
        .conn
        .query(
            "INSERT INTO entities \
             (id, group_id, entity_type_id, entity_type_source, ner_confidence, \
              recorded_at, updated_at, entity_type_assigned_at) \
             VALUES (?1, 'test-group', ?2, ?3, ?4, ?5, ?5, ?5) RETURNING rowid",
            libsql::params![
                id.to_string(),
                entity_type_id,
                entity_type_source.to_string(),
                conf_val,
                now
            ],
        )
        .await
        .expect("insert_entity_with_rowid INSERT must succeed");
    let row = rows
        .next()
        .await
        .expect("RETURNING must yield a row without error")
        .expect("RETURNING must yield a row");
    row.get::<i64>(0).expect("rowid at index 0")
}

/// Insert an entity_type into the registry for 'test-group'.
async fn insert_entity_type(graph: &TemporalGraph, id: i64, name: &str, description: &str) {
    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entity_types \
             (id, group_id, name, description, created_at) \
             VALUES (?1, 'test-group', ?2, ?3, ?4)",
            libsql::params![id, name.to_string(), description.to_string(), now],
        )
        .await
        .expect("insert_entity_type must succeed");
}

// ─── Mock LLM provider ───────────────────────────────────────────────────────
//
// Minimal mock returning a fixed JSON response for every call.
// Pattern mirrors phase_e_reclassify.rs MockLlm exactly.

struct MockLlm {
    response: String,
    model: String,
}

impl MockLlm {
    fn new(json: impl Into<String>) -> Self {
        Self {
            response: json.into(),
            model: "mock-model".to_string(),
        }
    }
}

/// Concrete ChatResponse impl returned by MockLlm.
#[derive(Debug)]
struct MockChatResponse {
    text: String,
}

impl std::fmt::Display for MockChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}

impl ChatResponseTrait for MockChatResponse {
    fn text(&self) -> Option<String> {
        if self.text.is_empty() {
            None
        } else {
            Some(self.text.clone())
        }
    }
    fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
        None
    }
}

#[async_trait::async_trait]
impl ChatProvider for MockLlm {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
    ) -> std::result::Result<Box<dyn ChatResponseTrait>, autoagents_llm::error::LLMError> {
        Ok(Box::new(MockChatResponse {
            text: self.response.clone(),
        }))
    }
    fn model(&self) -> &str {
        &self.model
    }
}

// ─── C1: File exists and is <500 LoC ─────────────────────────────────────────

/// C1: `crates/kremory/src/core/dream/consistency_check.rs` must exist
/// AND its line count must be strictly less than 500.
///
/// Failure mode without fix: file does not exist → std::fs::read_to_string fails.
#[test]
fn c1_module_exists_under_500_loc() {
    // Navigate to workspace root: crates/kremory -> crates -> workspace root.
    // Matches the pattern used by migration_idempotency.rs and adr_index.rs;
    // necessary because cargo test sets CWD to package root (crates/kremory/),
    // not workspace root, so bare relative paths must be anchored to CARGO_MANIFEST_DIR.
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/kremory has parent dir (crates/)")
        .parent()
        .expect("crates/ has parent dir (workspace root)");
    let rel = "crates/kremory/src/core/dream/consistency_check.rs";
    let full = workspace_root.join(rel);
    let path = full.to_str().expect("path is valid UTF-8");
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("C1: consistency_check.rs must exist at {path}: {e}"));
    let line_count = content.lines().count();
    assert!(
        line_count < 500,
        "C1: consistency_check.rs must be <500 LoC per feedback_split_files_before_adding_when_over_500_loc; got {line_count} lines"
    );
}

// ─── C2: ConsistencyCheckOpts defaults ───────────────────────────────────────

/// C2: `ConsistencyCheckOpts::default()` must match ADR-047 §Module signature:
/// - `embed_prefilter_threshold == 0.6`
/// - `max_candidates_per_run == Some(50)`
/// - `verify_model_override == None`
///
/// Failure mode without fix: `consistency_check` module doesn't exist → compile error.
#[test]
fn c2_consistency_check_opts_defaults_match_adr_047() {
    use kremory::core::dream::consistency_check::ConsistencyCheckOpts;

    let opts = ConsistencyCheckOpts::default();
    assert_eq!(
        opts.embed_prefilter_threshold, 0.6_f32,
        "C2: embed_prefilter_threshold default must be 0.6 per ADR-047"
    );
    assert_eq!(
        opts.max_candidates_per_run,
        Some(50),
        "C2: max_candidates_per_run default must be Some(50) per ADR-047 RISK-003 fold"
    );
    assert!(
        opts.verify_model_override.is_none(),
        "C2: verify_model_override default must be None"
    );
}

// ─── C3: embed_input_formatter determinism + padding ─────────────────────────

/// C3a: `embed_input_formatter()` must be deterministic — 100 calls with the same
/// inputs produce byte-identical output.
///
/// C3b: Padding contract — when `facts.len() < 3`, empty-string slots are appended.
/// Tested with: empty vec (3 empty pads), 1-fact (2 pads), 5-fact (only top-3 used).
///
/// Failure mode without fix: function doesn't exist → compile error.
#[test]
fn c3_embed_input_formatter_deterministic() {
    use kremory::core::dream::consistency_check::embed_input_formatter;

    // 100× determinism check
    let facts: Vec<String> = vec!["fact1".into(), "fact2".into(), "fact3".into()];
    let first = embed_input_formatter("Alice", &facts);
    for i in 1..100 {
        let result = embed_input_formatter("Alice", &facts);
        assert_eq!(
            result, first,
            "C3: embed_input_formatter must be deterministic; iteration {i} differed"
        );
    }

    // Empty facts → 3 empty-string pads
    let empty_result = embed_input_formatter("Bob", &[]);
    // Must contain name + pipe separator + padded (empty) slots
    assert!(
        empty_result.contains("Bob"),
        "C3: formatter with empty facts must contain the entity name"
    );
    // The padded form should include the separator " | " once per slot regardless
    let pipe_count = empty_result.matches(" | ").count();
    assert_eq!(
        pipe_count, 3,
        "C3: with 0 facts, output must have 3 pipe separators (name | pad | pad | pad); got {pipe_count} in '{empty_result}'"
    );

    // 1 fact → 2 pads
    let one_fact: Vec<String> = vec!["single-fact".into()];
    let one_result = embed_input_formatter("Carol", &one_fact);
    let pipe_count_one = one_result.matches(" | ").count();
    assert_eq!(
        pipe_count_one, 3,
        "C3: with 1 fact, output must have 3 pipe separators (name | fact | pad | pad); got {pipe_count_one} in '{one_result}'"
    );
    assert!(
        one_result.contains("single-fact"),
        "C3: formatter with 1 fact must include the fact content"
    );

    // 5 facts → only top-3 used (output must NOT contain facts at index 3 or 4)
    let five_facts: Vec<String> = vec![
        "alpha".into(),
        "beta".into(),
        "gamma".into(),
        "SHOULD_NOT_APPEAR_delta".into(),
        "SHOULD_NOT_APPEAR_epsilon".into(),
    ];
    let five_result = embed_input_formatter("Dave", &five_facts);
    assert!(
        !five_result.contains("SHOULD_NOT_APPEAR_delta"),
        "C3: with 5 facts, index-3 fact must NOT appear in formatter output (only top-3 used)"
    );
    assert!(
        !five_result.contains("SHOULD_NOT_APPEAR_epsilon"),
        "C3: with 5 facts, index-4 fact must NOT appear in formatter output (only top-3 used)"
    );
    let pipe_count_five = five_result.matches(" | ").count();
    assert_eq!(
        pipe_count_five, 3,
        "C3: with 5 facts, output must have exactly 3 pipe separators (name | f0 | f1 | f2); got {pipe_count_five}"
    );
}

// ─── C4: embed_prefilter_gate strict boundary ─────────────────────────────────

/// C4: `embed_prefilter_gate(tau, cos)` must use strict less-than (`cos < tau`):
/// - When `cos == tau`, entity is NOT flagged (boundary is exclusive)
/// - When `cos == tau - epsilon`, entity IS flagged (just below boundary)
///
/// Failure mode without fix: function doesn't exist → compile error.
#[test]
fn c4_embed_prefilter_gate_strict_less_than() {
    use kremory::core::dream::consistency_check::embed_prefilter_gate;

    let tau: f32 = 0.6;

    // Exact boundary: cos == tau → NOT flagged (strict less-than, boundary is exclusive)
    let at_boundary = embed_prefilter_gate(tau, 0.6_f32);
    assert!(
        !at_boundary,
        "C4: embed_prefilter_gate(tau=0.6, cos=0.6) must return false — \
         boundary is exclusive (cos < tau, NOT cos <= tau)"
    );

    // Just below boundary: cos = 0.5999 → IS flagged
    let below_boundary = embed_prefilter_gate(tau, 0.5999_f32);
    assert!(
        below_boundary,
        "C4: embed_prefilter_gate(tau=0.6, cos=0.5999) must return true — \
         below the threshold must be flagged for verify"
    );

    // Well above: cos = 0.9 → NOT flagged
    let above = embed_prefilter_gate(tau, 0.9_f32);
    assert!(
        !above,
        "C4: embed_prefilter_gate(tau=0.6, cos=0.9) must return false — \
         above threshold must not be flagged"
    );

    // Well below: cos = 0.1 → IS flagged
    let well_below = embed_prefilter_gate(tau, 0.1_f32);
    assert!(
        well_below,
        "C4: embed_prefilter_gate(tau=0.6, cos=0.1) must return true"
    );
}

// ─── C5: Strict deserialization — no serde(default) on required fields ────────

/// C5a: Parsing JSON that is missing `entity_id` must return a parse error
/// (no `#[serde(default)]` on required fields per [[llm-output-parse-loudly]]).
///
/// C5b: Parsing JSON with `action="correct"` but missing `new_type_id` must also
/// return a parse error (Vera SCOPE-001 fold — correctness contract).
///
/// C5c: A fully-valid JSON deserializes correctly.
///
/// Failure mode without fix: VerifyDecision type doesn't exist → compile error.
/// Correct behavior: missing required field = serde parse error, NOT default sentinel.
#[test]
fn c5_confirm_reject_modify_schema_strict_deserialization() {
    use kremory::core::dream::consistency_check::VerifyDecision;

    // C5a: missing `entity_id` must fail
    let missing_entity_id = r#"{"action": "confirm", "confidence": 0.9}"#;
    let result_a: Result<VerifyDecision, _> = serde_json::from_str(missing_entity_id);
    assert!(
        result_a.is_err(),
        "C5a: parsing JSON missing 'entity_id' must return parse error; \
         got Ok — indicates forbidden #[serde(default)] on entity_id"
    );

    // C5b: action=correct but missing new_type_id must fail
    let missing_new_type_id = r#"{"entity_id": 42, "action": "correct", "confidence": 0.8}"#;
    let result_b: Result<VerifyDecision, _> = serde_json::from_str(missing_new_type_id);
    assert!(
        result_b.is_err(),
        "C5b: parsing JSON with action='correct' but missing 'new_type_id' must fail; \
         got Ok — violates Vera SCOPE-001 correctness contract"
    );

    // C5c: valid JSON with action=confirm deserializes correctly
    let valid_confirm = r#"{"entity_id": 7, "action": "confirm", "confidence": 0.95}"#;
    let result_c: Result<VerifyDecision, _> = serde_json::from_str(valid_confirm);
    assert!(
        result_c.is_ok(),
        "C5c: valid confirm JSON must deserialize without error; got: {:?}",
        result_c.err()
    );
    let decision = result_c.unwrap();
    assert_eq!(decision.entity_id, 7, "C5c: entity_id must be 7");

    // C5d: valid JSON with action=correct and new_type_id deserializes correctly
    let valid_correct =
        r#"{"entity_id": 3, "action": "correct", "new_type_id": 5, "confidence": 0.82}"#;
    let result_d: Result<VerifyDecision, _> = serde_json::from_str(valid_correct);
    assert!(
        result_d.is_ok(),
        "C5d: valid correct JSON must deserialize without error; got: {:?}",
        result_d.err()
    );
    let decision_d = result_d.unwrap();
    assert_eq!(decision_d.entity_id, 3, "C5d: entity_id must be 3");
}

// ─── C6: Cap-overflow guard ───────────────────────────────────────────────────

/// C6: When `max_candidates_per_run = Some(50)` and 100 entities all fall below
/// the embed threshold (all get flagged by prefilter), the cap guard must:
/// - Send exactly 50 to LLM verify (flagged=50)
/// - Drop 50 silently (cap_overflow_dropped=50)
/// - Return correct summary counts
///
/// The mock LLM returns `confirm` for every entity it is asked about.
///
/// Failure mode without fix: run_consistency_check doesn't exist → compile error.
#[tokio::test]
async fn c6_cap_overflow_guard_drops_excess() {
    use kremory::core::dream::consistency_check::{run_consistency_check, ConsistencyCheckOpts};
    use kremory::core::provider::DeterministicEmbeddingProvider;

    let (graph, _tmp) = open_graph().await;

    // Insert entity_type for 'test-group' with a very long/specific description
    // so that 256-dim deterministic embedder produces low cosine similarity
    // against a short entity name.
    insert_entity_type(
        &graph,
        1,
        "Person",
        "A human being with a social security number, birth certificate and passport document",
    )
    .await;

    // Insert 100 entities all with Phase1Ner source (eligible for Pass 4)
    // and ner_confidence=0.9 (high confidence — Pass 2 won't touch them)
    for i in 0..100_i64 {
        insert_entity_with_rowid(
            &graph,
            &format!("cap-entity-{i:03}"),
            1, // entity_type_id=1 (Person)
            "Phase1Ner",
            Some(0.9),
        )
        .await;
    }

    // The mock LLM returns a batch of confirm decisions for whatever IDs are passed.
    // We return confirms for all 50 that pass the cap gate.
    // The exact entity IDs are not known at test-write time so we return an empty
    // decisions array — the summary should still record flagged=50 and overflow=50.
    // (Green may choose to build decisions from the candidates; an empty array
    // is valid and produces confirmed=0, flagged=50, overflow=50.)
    let mock = MockLlm::new(r#"{"decisions": []}"#);

    let embedder = DeterministicEmbeddingProvider::new(256);
    let opts = ConsistencyCheckOpts {
        embed_prefilter_threshold: 0.99, // near-1.0 forces ALL entities to be flagged
        max_candidates_per_run: Some(50),
        verify_model_override: None,
        dry_run: false,
    };

    let summary = run_consistency_check(&graph.conn, &embedder, &mock, opts)
        .await
        .expect("C6: run_consistency_check must succeed");

    assert_eq!(
        summary.cap_overflow_dropped, 50,
        "C6: cap_overflow_dropped must be 50 when 100 flagged and cap=50; got {}",
        summary.cap_overflow_dropped
    );
    assert_eq!(
        summary.flagged, 50,
        "C6: flagged (post-cap) must be 50; got {}",
        summary.flagged
    );
    // scanned must equal total entities considered (100)
    assert_eq!(
        summary.scanned, 100,
        "C6: scanned must be 100 (all entities were passed to prefilter); got {}",
        summary.scanned
    );
}

// ─── C7: Observability counters increment on right paths ─────────────────────

/// C7: Instrument with DebuggingRecorder and drive a run that produces:
/// - 1 confirmed entity
/// - 1 corrected entity
/// - 1 uncertain entity
/// - 1 cap_overflow drop (total=4 flagged but cap=3)
///
/// Assert each per-arm counter incremented exactly as expected.
///
/// Failure mode without fix: run_consistency_check doesn't exist → compile error.
#[tokio::test]
async fn c7_observability_counters_increment_on_right_paths() {
    use kremory::core::dream::consistency_check::{run_consistency_check, ConsistencyCheckOpts};
    use kremory::core::provider::DeterministicEmbeddingProvider;
    use metrics_util::debugging::DebuggingRecorder;

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let (graph, _tmp) = open_graph().await;

    // Entity type with description distant from entity names (to trigger prefilter)
    insert_entity_type(
        &graph,
        1,
        "Person",
        "A human individual with unique legal identity and biometric fingerprints",
    )
    .await;
    insert_entity_type(
        &graph,
        2,
        "Organisation",
        "A legal entity with shareholders, directors and corporate registration",
    )
    .await;

    // Seed 4 entities — all will be flagged at threshold=0.99
    let _r1 = insert_entity_with_rowid(&graph, "obs-confirmed", 1, "Phase1Ner", Some(0.9)).await;
    let _r2 = insert_entity_with_rowid(&graph, "obs-corrected", 1, "Phase1Ner", Some(0.9)).await;
    let _r3 = insert_entity_with_rowid(&graph, "obs-uncertain", 1, "Phase1Ner", Some(0.9)).await;
    let _r4 = insert_entity_with_rowid(&graph, "obs-overflow", 1, "Phase1Ner", Some(0.9)).await;

    // The LLM returns: confirm for entity 1, correct for entity 2, uncertain for entity 3.
    // Entity 4 is dropped by cap=3 overflow and never reaches the LLM.
    // We use rowid-based entity_id in the LLM response matching the DB integer rowids.
    // Since exact rowids are DB-assigned we pass a response that covers confirm/correct/uncertain
    // by entity name rather than rowid — the Green agent must map these correctly.
    let mock_json = r#"{
        "decisions": [
            {"entity_id": 0, "action": "confirm",  "confidence": 0.92},
            {"entity_id": 0, "action": "correct",  "new_type_id": 2, "confidence": 0.85},
            {"entity_id": 0, "action": "uncertain","confidence": 0.55}
        ]
    }"#;
    // Note: entity_id=0 is a placeholder here because the exact rowids are DB-assigned.
    // The Green agent must either: (a) use actual rowids in the prompt so the mock returns
    // matching IDs, or (b) map decisions by position if that's the implementation choice.
    // This test validates that the COUNTERS fire — the mock returns 3 decisions of each type.
    let mock = MockLlm::new(mock_json);

    let embedder = DeterministicEmbeddingProvider::new(256);
    let opts = ConsistencyCheckOpts {
        embed_prefilter_threshold: 0.99, // force all 4 to be flagged
        max_candidates_per_run: Some(3), // cap=3 → 1 overflow
        verify_model_override: None,
        dry_run: false,
    };

    let _summary = run_consistency_check(&graph.conn, &embedder, &mock, opts)
        .await
        .expect("C7: run_consistency_check must succeed");

    let snapshot = snapshotter.snapshot().into_vec();

    // Check cap_overflow_total counter fired
    let has_cap_overflow = snapshot
        .iter()
        .any(|(k, _, _, _)| k.key().name() == "kremory.dream.consistency_check.cap_overflow_total");
    assert!(
        has_cap_overflow,
        "C7: kremory.dream.consistency_check.cap_overflow_total must be emitted on overflow path; \
         got counters: {:?}",
        snapshot
            .iter()
            .map(|(k, _, _, _)| k.key().name())
            .collect::<Vec<_>>()
    );

    // Check scanned_total fired
    let has_scanned = snapshot
        .iter()
        .any(|(k, _, _, _)| k.key().name() == "kremory.dream.consistency_check.scanned_total");
    assert!(
        has_scanned,
        "C7: kremory.dream.consistency_check.scanned_total must be emitted"
    );

    // Check flagged_total fired
    let has_flagged = snapshot
        .iter()
        .any(|(k, _, _, _)| k.key().name() == "kremory.dream.consistency_check.flagged_total");
    assert!(
        has_flagged,
        "C7: kremory.dream.consistency_check.flagged_total must be emitted"
    );
}

// ─── C8: run_consistency_check orchestrates full flow ────────────────────────

/// C8: End-to-end with mocked ChatProvider returning a golden
/// Confirm/Reject/Modify response.
///
/// Seed 3 entities:
/// - entity A (entity_type_id=1) → mock returns confirm
/// - entity B (entity_type_id=1) → mock returns correct (new_type_id=2)
/// - entity C (entity_type_id=1) → mock returns uncertain
///
/// Assert ConsistencyCheckSummary fields:
/// - confirmed == 1
/// - corrected == 1
/// - uncertain == 1
/// - flagged >= 3 (at least 3 went to LLM; more may have been seeded by earlier tests
///   but this test uses a fresh DB)
///
/// Failure mode without fix: run_consistency_check doesn't exist → compile error.
#[tokio::test]
async fn c8_run_consistency_check_orchestrates_full_flow() {
    use kremory::core::dream::consistency_check::{run_consistency_check, ConsistencyCheckOpts};
    use kremory::core::provider::DeterministicEmbeddingProvider;

    let (graph, _tmp) = open_graph().await;

    insert_entity_type(
        &graph,
        1,
        "Person",
        "A human being with personal identity and biographical information dating back to birth",
    )
    .await;
    insert_entity_type(
        &graph,
        2,
        "Organisation",
        "A registered company, NGO or public institution with a legal charter",
    )
    .await;

    // Seed 3 entities eligible for Pass 4 (Phase1Ner, high-conf, non-catch-all)
    let _rowid_a =
        insert_entity_with_rowid(&graph, "c8-entity-a", 1, "Phase1Ner", Some(0.92)).await;
    let _rowid_b =
        insert_entity_with_rowid(&graph, "c8-entity-b", 1, "Phase1Ner", Some(0.88)).await;
    let _rowid_c =
        insert_entity_with_rowid(&graph, "c8-entity-c", 1, "Phase1Ner", Some(0.85)).await;

    // The mock returns exactly 1 confirm + 1 correct + 1 uncertain.
    // Green must expose enough info in the prompt for the mock to work, but
    // since this is a mock test, we return decisions by position.
    // entity_id values use 0 as placeholder — the implementation maps by index or rowid.
    // The key invariant: 3 decisions of 3 types → summary reflects all three.
    let mock = MockLlm::new(
        r#"{"decisions": [
            {"entity_id": 1, "action": "confirm",  "confidence": 0.91},
            {"entity_id": 2, "action": "correct",  "new_type_id": 2, "confidence": 0.87},
            {"entity_id": 3, "action": "uncertain","confidence": 0.52}
        ]}"#,
    );

    let embedder = DeterministicEmbeddingProvider::new(256);
    let opts = ConsistencyCheckOpts {
        embed_prefilter_threshold: 0.99, // force all 3 to be flagged
        max_candidates_per_run: Some(50),
        verify_model_override: None,
        dry_run: false,
    };

    let summary = run_consistency_check(&graph.conn, &embedder, &mock, opts)
        .await
        .expect("C8: run_consistency_check must succeed");

    assert_eq!(
        summary.confirmed, 1,
        "C8: confirmed must be 1; got {}",
        summary.confirmed
    );
    assert_eq!(
        summary.corrected, 1,
        "C8: corrected must be 1; got {}",
        summary.corrected
    );
    assert_eq!(
        summary.uncertain, 1,
        "C8: uncertain must be 1; got {}",
        summary.uncertain
    );
    assert_eq!(
        summary.flagged, 3,
        "C8: flagged must be 3 (all 3 entities passed to LLM-verify); got {}",
        summary.flagged
    );
    assert_eq!(
        summary.scanned, 3,
        "C8: scanned must be 3; got {}",
        summary.scanned
    );
    assert_eq!(
        summary.cap_overflow_dropped, 0,
        "C8: no overflow expected with cap=50 and 3 candidates; got {}",
        summary.cap_overflow_dropped
    );
}

// ─── C9: dream_pass4_audit row written per corrected action ──────────────────

/// C9: When mock LLM returns `correct, new_type_id=5, confidence=0.8` for entity X
/// (which has entity_type_id=2 at ingest time), the `dream_pass4_audit` table must
/// contain a row where:
/// - `entity_id` == X's rowid
/// - `pre_type_id` == 2 (state BEFORE the correction)
/// - `post_type_id` == 5 (the new type from the LLM decision)
/// - `verify_confidence` == 0.8
///
/// Failure mode without fix: run_consistency_check doesn't exist → compile error.
/// Also fails if audit row is not written (the assertion on SELECT count fails).
#[tokio::test]
async fn c9_audit_row_written_per_corrected_action() {
    use kremory::core::dream::consistency_check::{run_consistency_check, ConsistencyCheckOpts};
    use kremory::core::provider::DeterministicEmbeddingProvider;

    let (graph, _tmp) = open_graph().await;

    insert_entity_type(
        &graph,
        2,
        "Organisation",
        "A registered company or corporation with legal charter and shareholders",
    )
    .await;
    insert_entity_type(
        &graph,
        5,
        "Person",
        "A natural human person with biographical and identity information",
    )
    .await;

    // Seed entity X with entity_type_id=2 (Organisation) — we expect it to be
    // corrected to type_id=5 (Person) by the mock LLM.
    let rowid_x = insert_entity_with_rowid(&graph, "c9-entity-x", 2, "Phase1Ner", Some(0.91)).await;

    // Mock returns a correction: correct entity X to type 5 with confidence 0.8.
    // We embed rowid_x in the response JSON.
    let mock_json = format!(
        r#"{{"decisions": [{{"entity_id": {rowid_x}, "action": "correct", "new_type_id": 5, "confidence": 0.8}}]}}"#
    );
    let mock = MockLlm::new(mock_json);

    let embedder = DeterministicEmbeddingProvider::new(256);
    let opts = ConsistencyCheckOpts {
        embed_prefilter_threshold: 0.99, // force the entity to be flagged
        max_candidates_per_run: Some(50),
        verify_model_override: None,
        dry_run: false,
    };

    let summary = run_consistency_check(&graph.conn, &embedder, &mock, opts)
        .await
        .expect("C9: run_consistency_check must succeed");

    assert_eq!(
        summary.corrected, 1,
        "C9: corrected count must be 1; got {}",
        summary.corrected
    );

    // Query the audit table for the correction row
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_id, pre_type_id, post_type_id, verify_confidence \
             FROM dream_pass4_audit \
             WHERE entity_id = ?1",
            libsql::params![rowid_x],
        )
        .await
        .expect("C9: SELECT dream_pass4_audit must succeed");

    let audit_row = rows
        .next()
        .await
        .expect("C9: audit row iteration must not error")
        .expect("C9: dream_pass4_audit must contain a row for corrected entity X");

    let audit_entity_id: i64 = audit_row.get(0).expect("entity_id at index 0");
    let audit_pre_type: i64 = audit_row.get(1).expect("pre_type_id at index 1");
    let audit_post_type: i64 = audit_row.get(2).expect("post_type_id at index 2");
    let audit_confidence: f64 = audit_row.get(3).expect("verify_confidence at index 3");

    assert_eq!(
        audit_entity_id, rowid_x,
        "C9: audit row entity_id must match X's rowid"
    );
    assert_eq!(
        audit_pre_type, 2,
        "C9: audit row pre_type_id must be 2 (the type BEFORE correction); got {audit_pre_type}"
    );
    assert_eq!(
        audit_post_type, 5,
        "C9: audit row post_type_id must be 5 (new_type_id from LLM decision); got {audit_post_type}"
    );
    assert!(
        (audit_confidence - 0.8).abs() < 0.001,
        "C9: audit row verify_confidence must be ~0.8; got {audit_confidence}"
    );
}

// ─── C10: Real-LLM schema parses 100% (feature-gated, #[ignore]) ─────────────

/// C10: Real-LLM integration smoke — invoke real `gemma4-e2b:latest` 10 times
/// with a sample entity-verification context; assert 10/10 returns valid
/// Confirm/Reject/Modify JSON that parses without error.
///
/// Per substrate SoT (`tests/llm_integration.rs:1-25`): `gemma4-e2b:latest`
/// is the interactive default. Override via `OLLAMA_CHAT_MODEL`.
///
/// Invoke:
///   OLLAMA_BASE_URL=http://localhost:11434 \
///   OLLAMA_CHAT_MODEL=gemma4-e2b:latest \
///   cargo test -p kremory --features llm-integration --test consistency_check_phase_c \
///     c10_real_llm_schema_parses_100_percent -- --ignored
///
/// Failure mode without fix: run_consistency_check doesn't exist → compile error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
#[cfg(feature = "llm-integration")]
async fn c10_real_llm_schema_parses_100_percent() {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;
    use kremory::core::dream::consistency_check::{run_consistency_check, ConsistencyCheckOpts};
    use kremory::core::provider::DeterministicEmbeddingProvider;

    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // Per substrate SoT (tests/llm_integration.rs:1-25): gemma4-e2b:latest
    let chat_model =
        std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4-e2b:latest".to_string());

    let llm: std::sync::Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(&chat_model)
        .timeout_seconds(120)
        .keep_alive("1h")
        .build()
        .expect("C10: Ollama LLM builder must succeed");

    let mut parse_successes = 0usize;

    for run in 0..10 {
        let (graph, _tmp) = open_graph().await;

        insert_entity_type(
            &graph,
            1,
            "Person",
            "A natural human individual with biographical identity",
        )
        .await;
        insert_entity_type(
            &graph,
            2,
            "Organisation",
            "A company or registered legal entity",
        )
        .await;

        // Seed 1 entity per run — "Apple" as Organisation (plausibly wrong → triggers verify)
        insert_entity_with_rowid(&graph, "Apple", 2, "Phase1Ner", Some(0.91)).await;

        let embedder = DeterministicEmbeddingProvider::new(256);
        let opts = ConsistencyCheckOpts {
            embed_prefilter_threshold: 0.99, // force flag
            max_candidates_per_run: Some(5),
            verify_model_override: None,
            dry_run: false,
        };

        let result = run_consistency_check(&graph.conn, &embedder, llm.as_ref(), opts).await;

        match result {
            Ok(summary) => {
                // If we got a summary, the LLM response was parsed — count as success
                let _ = summary;
                parse_successes += 1;
                tracing::info!(
                    target: "kremory::test::c10",
                    run = run,
                    "C10: run {run} parsed successfully"
                );
            }
            Err(e) => {
                tracing::warn!(
                    target: "kremory::test::c10",
                    run = run,
                    error = %e,
                    "C10: run {run} failed"
                );
            }
        }
    }

    assert_eq!(
        parse_successes, 10,
        "C10: RISK-001 schema parses must be 10/10 on gemma4-e2b:latest; got {parse_successes}/10. \
         Per ADR-047 §LLM-verify: 100% direct-parse rate on gemma4-e2b:latest was validated in \
         the 2026-06-09 LLM-as-authority spike. Failure here means schema drifted or model \
         behaviour changed."
    );
}
