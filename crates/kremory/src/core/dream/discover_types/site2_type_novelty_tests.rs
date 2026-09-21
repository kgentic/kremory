// ─── Site #2 — type-novelty LLM-verify band ─────────────────────────────────
//
// Deterministic tests for the `DreamOpts::include_type_novelty_llm_verify` flag:
// (a) flag OFF preserves the pre-Site-#2 outcome exactly (regression guard); (b)
// flag ON + LLM says "same" → proposal rejected as redundant; (c) flag ON + LLM
// says "distinct" → proposal accepted (the EDC false-reject-prevention case this
// site exists to fix). Mirrors `type_registry_collapse.rs`'s
// `ScriptedVerdictProvider` pattern, but `discover_types` makes TWO sequential LLM
// calls when the LLM-verify band engages (1: the discovery proposal call, 2: the
// Site #2 adjudication call) — `ScriptedSequenceProvider` returns one scripted
// response per call, in order.

use super::*;
use crate::core::entity_types::ensure_default_types_seeded;
use crate::core::provider::{
    ChatMessage, ChatResponse, LLMError, MockChatResponse, StructuredOutputFormat, Tool,
};
use crate::core::schema::TemporalGraph;
use std::sync::Mutex;

/// Scripted `ChatProvider` returning one fixed JSON response PER CALL, in
/// order (first call gets `responses[0]`, second gets `responses[1]`, ...).
/// Panics if called more times than responses are scripted — a test-shape
/// bug, not a production concern.
#[derive(Debug)]
struct ScriptedSequenceProvider {
    responses: Mutex<std::collections::VecDeque<String>>,
}

impl ScriptedSequenceProvider {
    fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().map(String::from).collect()),
        }
    }
}

#[async_trait::async_trait]
impl ChatProvider for ScriptedSequenceProvider {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
        let mut queue = self.responses.lock().expect("mutex poisoned");
        let text = queue
            .pop_front()
            .expect("ScriptedSequenceProvider: called more times than scripted responses");
        Ok(Box::new(MockChatResponse { text }))
    }
}

/// Seed a group with the default 0..=9 types plus ONE custom existing type
/// (id=11) whose description embeds to `unit_vec4(1,0,0,0)` — used as the
/// Site #2 candidate's `existing_name`/`existing_desc`.
#[allow(clippy::too_many_arguments)] // test helper — test files are exempt from the arg-count lint
async fn seed_existing_type(conn: &libsql::Connection, group_id: &str, name: &str, desc: &str) {
    ensure_default_types_seeded(conn, group_id)
        .await
        .expect("seed defaults 0..=9");
    conn.execute(
        "INSERT INTO entity_types (group_id, id, name, description) VALUES (?1, 11, ?2, ?3)",
        libsql::params![group_id, name, desc],
    )
    .await
    .expect("insert existing type");
}

/// Deterministic embedder: pre-registered vector per input text (exact
/// match), zero vector otherwise (mirrors `type_registry_collapse.rs`'s
/// `MockEmbeddingProvider`).
#[derive(Debug, Clone)]
struct MockEmbeddingProvider {
    vectors: std::collections::HashMap<String, Vec<f32>>,
}

impl DynEmbeddingProvider for MockEmbeddingProvider {
    fn embed_dyn<'a>(
        &'a self,
        text: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>>> + Send + 'a>>
    {
        let v = self
            .vectors
            .get(text)
            .cloned()
            .unwrap_or_else(|| vec![0.0_f32; 4]);
        Box::pin(async move { Ok(v) })
    }
    fn last_usage_tokens_dyn(&self) -> Option<u64> {
        None
    }
}

/// (a) Flag OFF (default): a proposal whose desc-cosine is ≥0.85 against an
/// existing type with ZERO lemma overlap (the `NeedsLlmVerify` case) is
/// REJECTED — the exact pre-Site-#2 outcome (hard cutoff, no LLM adjudication
/// call made). Only ONE LLM call is scripted (the discovery proposal call) —
/// if the flag wrongly engaged Site #2 adjudication, `ScriptedSequenceProvider`
/// would panic on a second call, proving no extra call was made.
#[tokio::test]
async fn flag_off_preserves_pre_site2_hard_cutoff_reject() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let conn = graph.conn.clone();
    // "Individual" (NOT the seeded default "Person", id=1 — avoid a UNIQUE
    // collision) — zero lemma overlap with the proposed "Human".
    seed_existing_type(
        &conn,
        "g1",
        "Individual",
        "A living individual, described by name and biography.",
    )
    .await;

    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, 0, ?2, ?3)",
        libsql::params!["alice smith", now, "g1"],
    )
    .await
    .expect("insert catch-all entity");

    // Proposal "Human" desc embeds identically to "Individual" desc → cosine 1.0.
    let shared_vec = vec![1.0f32, 0.0, 0.0, 0.0];
    let mut vectors = std::collections::HashMap::new();
    vectors.insert(
        "A living individual, described by name and biography.".to_string(),
        shared_vec.clone(),
    );
    vectors.insert(
        "A person, described by name and biography facts.".to_string(),
        shared_vec,
    );
    let embedder = MockEmbeddingProvider { vectors };

    // Only ONE response scripted — the discovery proposal call. No second
    // (adjudication) call should ever be made with the flag off.
    let llm = ScriptedSequenceProvider::new(vec![
        r#"{"proposals":[{"name":"Human","description":"A person, described by name and biography facts.","justification":"catch-all evidence"}]}"#,
    ]);

    let result = discover_types(
        &llm,
        DiscoverTypesParams {
            conn: &conn,
            group_id: "g1",
            embedder: Some(&embedder),
            max_proposals: 3,
            model_id: "test-model",
            llm_verify_band: false,
            evidence_retype_by_similarity: false,
        },
    )
    .await
    .expect("discover_types must succeed");

    assert!(
        result.types_accepted.is_empty(),
        "flag OFF: NeedsLlmVerify at >=0.85 must reject (pre-Site-#2 hard cutoff), got accepted={:?}",
        result.types_accepted
    );
    assert_eq!(result.types_rejected.len(), 1, "one rejection");
    assert!(
        result.types_rejected[0].1.starts_with("redundant_with:"),
        "rejection reason must be redundant_with:, got {:?}",
        result.types_rejected[0].1
    );
}

/// (a') Flag OFF: a proposal in the [0.70, 0.85) ambiguous band is ACCEPTED —
/// the pre-Site-#2 "no candidate" outcome (there was no ambiguous-band concept
/// before Site #2; anything below 0.85 simply passed).
#[tokio::test]
async fn flag_off_mid_band_accepts() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let conn = graph.conn.clone();
    seed_existing_type(
        &conn,
        "g2",
        "Vehicle",
        "A car, truck, or other conveyance used for transport.",
    )
    .await;

    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, 0, ?2, ?3)",
        libsql::params!["red wagon", now, "g2"],
    )
    .await
    .expect("insert catch-all entity");

    // Cosine 0.80 (mid-band): A=[1,0,0,0], B=[0.8,0.6,0,0].
    let mut vectors = std::collections::HashMap::new();
    vectors.insert(
        "A car, truck, or other conveyance used for transport.".to_string(),
        vec![1.0f32, 0.0, 0.0, 0.0],
    );
    vectors.insert(
        "A wheeled conveyance pulled or pushed by hand.".to_string(),
        vec![0.8f32, 0.6, 0.0, 0.0],
    );
    let embedder = MockEmbeddingProvider { vectors };

    let llm = ScriptedSequenceProvider::new(vec![
        r#"{"proposals":[{"name":"Conveyance","description":"A wheeled conveyance pulled or pushed by hand.","justification":"catch-all evidence"}]}"#,
    ]);

    let result = discover_types(
        &llm,
        DiscoverTypesParams {
            conn: &conn,
            group_id: "g2",
            embedder: Some(&embedder),
            max_proposals: 3,
            model_id: "test-model",
            llm_verify_band: false,
            evidence_retype_by_similarity: false,
        },
    )
    .await
    .expect("discover_types must succeed");

    assert_eq!(
        result.types_accepted.len(),
        1,
        "flag OFF: mid-band [0.70,0.85) must accept (pre-Site-#2 'no candidate' outcome), rejected={:?}",
        result.types_rejected
    );
}

/// (b) Flag ON + scripted LLM `is_same_entity=true` (high confidence) →
/// `type_novelty_is_redundant` → the proposal IS redundant →
/// REJECTED. (The lemma overlap between "Organizations"/"Organization" is now
/// irrelevant to the decision — it mattered only to the OLD write_gate Row 5;
/// the confident `true` verdict alone is decisive.) Mid-band
/// cosine (0.80, in `[0.70, 0.85)`) is used so `check_proposal` nominates
/// `NeedsLlmVerify` rather than auto-rejecting at the pure classification step
/// (lemma overlap at cosine ≥0.85 would short-circuit to `Redundant` before
/// any LLM call).
#[tokio::test]
async fn flag_on_llm_says_same_with_lemma_signal_rejects_proposal() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let conn = graph.conn.clone();
    seed_existing_type(
        &conn,
        "g3",
        "Organization",
        "A group of people with a shared purpose or structure.",
    )
    .await;

    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, 0, ?2, ?3)",
        libsql::params!["acme co", now, "g3"],
    )
    .await
    .expect("insert catch-all entity");

    // Cosine 0.80 (mid-band): A=[1,0,0,0], B=[0.8,0.6,0,0].
    let mut vectors = std::collections::HashMap::new();
    vectors.insert(
        "A group of people with a shared purpose or structure.".to_string(),
        vec![1.0f32, 0.0, 0.0, 0.0],
    );
    vectors.insert(
        "A collective body of people organised for a common purpose.".to_string(),
        vec![0.8f32, 0.6, 0.0, 0.0],
    );
    let embedder = MockEmbeddingProvider { vectors };

    // Call 1: discovery proposal "Organizations" (trailing-s lemma match with
    // "Organization" → deterministic signal fires). Call 2: Site #2
    // adjudication → LLM says same.
    let llm = ScriptedSequenceProvider::new(vec![
        r#"{"proposals":[{"name":"Organizations","description":"A collective body of people organised for a common purpose.","justification":"catch-all evidence"}]}"#,
        r#"{"verdicts":[{"pair_id":0,"is_same_entity":true,"confidence":0.95,"reasoning":"same concept as Organization"}]}"#,
    ]);

    let result = discover_types(
        &llm,
        DiscoverTypesParams {
            conn: &conn,
            group_id: "g3",
            embedder: Some(&embedder),
            max_proposals: 3,
            model_id: "test-model",
            llm_verify_band: true,
            evidence_retype_by_similarity: false,
        },
    )
    .await
    .expect("discover_types must succeed");

    assert!(
        result.types_accepted.is_empty(),
        "flag ON + confident LLM true verdict: proposal must be rejected as \
         redundant (type_novelty_is_redundant), got accepted={:?}",
        result.types_accepted
    );
    assert_eq!(result.types_rejected.len(), 1);
    assert!(result.types_rejected[0].1.starts_with("redundant_with:"));
}

/// (b') Flag ON + scripted LLM `is_same_entity=true` (high confidence) but
/// WITHOUT any deterministic lemma signal — the exact Site #2 bug this
/// fixes. Under the OLD shared `write_gate` this hit Row 6 (no deterministic
/// corroboration → `PotentialAlias` → accept → DUPLICATE type). Under
/// `type_novelty_is_redundant`, a confident `true` verdict is the
/// terminal arbiter (type synonyms are lexically dissimilar by nature) → the
/// proposal IS redundant → REJECTED. This assertion FLIPPED with the fix.
#[tokio::test]
async fn flag_on_llm_says_same_without_lemma_signal_now_rejects_redundant() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let conn = graph.conn.clone();
    seed_existing_type(
        &conn,
        "g5",
        "Individual",
        "A living individual, described by name and biography.",
    )
    .await;

    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, 0, ?2, ?3)",
        libsql::params!["bob jones", now, "g5"],
    )
    .await
    .expect("insert catch-all entity");

    let shared_vec = vec![1.0f32, 0.0, 0.0, 0.0];
    let mut vectors = std::collections::HashMap::new();
    vectors.insert(
        "A living individual, described by name and biography.".to_string(),
        shared_vec.clone(),
    );
    vectors.insert(
        "A person, described by name and biography facts.".to_string(),
        shared_vec,
    );
    let embedder = MockEmbeddingProvider { vectors };

    // "Human"/"Individual" share zero lemma overlap → no deterministic signal.
    let llm = ScriptedSequenceProvider::new(vec![
        r#"{"proposals":[{"name":"Human","description":"A person, described by name and biography facts.","justification":"catch-all evidence"}]}"#,
        r#"{"verdicts":[{"pair_id":0,"is_same_entity":true,"confidence":0.95,"reasoning":"same concept as Individual"}]}"#,
    ]);

    let result = discover_types(
        &llm,
        DiscoverTypesParams {
            conn: &conn,
            group_id: "g5",
            embedder: Some(&embedder),
            max_proposals: 3,
            model_id: "test-model",
            llm_verify_band: true,
            evidence_retype_by_similarity: false,
        },
    )
    .await
    .expect("discover_types must succeed");

    assert!(
        result.types_accepted.is_empty(),
        "Confident LLM `is_same=true` verdict (no lemma signal needed) → \
         redundant → REJECT (was accept under write_gate Row 6), accepted={:?}",
        result.types_accepted
    );
    assert_eq!(result.types_rejected.len(), 1);
    assert!(result.types_rejected[0].1.starts_with("redundant_with:"));
}

/// (c) Flag ON + scripted LLM `is_same_entity=false` → the proposal is
/// DISTINCT → ACCEPTED. This is the EDC false-reject-prevention case Site #2
/// exists to fix: a hard 0.85 cutoff alone would have rejected this proposal
/// (desc-cosine 1.0, zero lemma overlap with "Person"), but the LLM correctly
/// distinguishes "Human" (a person, generically) from "Person" (an existing
/// registered type) as intended for this fixture — the LLM verdict is honored.
#[tokio::test]
async fn flag_on_llm_says_distinct_accepts_proposal() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let conn = graph.conn.clone();
    seed_existing_type(
        &conn,
        "g4",
        "LegalRuling",
        "A court's official decision on a case.",
    )
    .await;

    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, 0, ?2, ?3)",
        libsql::params!["marbury v madison", now, "g4"],
    )
    .await
    .expect("insert catch-all entity");

    let shared_vec = vec![1.0f32, 0.0, 0.0, 0.0];
    let mut vectors = std::collections::HashMap::new();
    vectors.insert(
        "A court's official decision on a case.".to_string(),
        shared_vec.clone(),
    );
    vectors.insert(
        "A prior court decision used as precedent for future cases.".to_string(),
        shared_vec,
    );
    let embedder = MockEmbeddingProvider { vectors };

    // Call 1: discovery proposal "LegalPrecedent" (desc-cosine 1.0 vs
    // "LegalRuling", zero lemma overlap → NeedsLlmVerify). Call 2: Site #2
    // adjudication → LLM correctly says DISTINCT.
    let llm = ScriptedSequenceProvider::new(vec![
        r#"{"proposals":[{"name":"LegalPrecedent","description":"A prior court decision used as precedent for future cases.","justification":"catch-all evidence"}]}"#,
        r#"{"verdicts":[{"pair_id":0,"is_same_entity":false,"confidence":0.9,"reasoning":"precedent and ruling are related but distinct legal concepts"}]}"#,
    ]);

    let result = discover_types(
        &llm,
        DiscoverTypesParams {
            conn: &conn,
            group_id: "g4",
            embedder: Some(&embedder),
            max_proposals: 3,
            model_id: "test-model",
            llm_verify_band: true,
            evidence_retype_by_similarity: false,
        },
    )
    .await
    .expect("discover_types must succeed");

    assert_eq!(
        result.types_accepted.len(),
        1,
        "flag ON + LLM false verdict: distinct-but-similar type must be accepted \
         (EDC false-reject-prevention case), rejected={:?}",
        result.types_rejected
    );
    assert_eq!(result.types_accepted[0].name, "LegalPrecedent");
}

/// The flag-OFF, no-LLM-call, ACCEPT-via-`Pass` path is exactly
/// the branch that used to leave ZERO evidence: prior
/// to this fix, `discover_types` never wrote to `identity_verdict_audit`
/// at all on `GateOutcome::Pass`, so an accepted proposal's `desc_cosine`
/// (the value that decided it was novel enough to accept) could not be
/// reconstructed from stored state after the fact — only re-embedding the
/// stored descriptions live could recover it ("the flag-on counterfactual
/// CANNOT be read from stored state").
///
/// One existing type is seeded so a REAL comparison happens (best_cosine
/// is computed, not skipped) — and the proposal's description embeds
/// ORTHOGONALLY to it, so cosine ~0.0, well below `TYPE_NOVELTY_LOWER_BAND`
/// (0.70) → `GateOutcome::Pass` with `existing_name = Some`, `desc_cosine
/// = Some(0.0)`. The assertion is specifically that the persisted cosine
/// is NON-NULL — proving the audit row records a REAL comparison, not the
/// registry-empty `None` case (that's covered by
/// `check_proposal`'s own `gate_passes_with_empty_existing_types` unit
/// test in `anti_redundancy.rs`).
#[tokio::test]
async fn accepted_proposal_with_existing_type_leaves_audit_row_with_cosine() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let conn = graph.conn.clone();
    seed_existing_type(
        &conn,
        "g_audit",
        "WeatherEvent",
        "A meteorological occurrence such as a storm or heatwave.",
    )
    .await;

    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, 0, ?2, ?3)",
        libsql::params!["some catch-all evidence", now, "g_audit"],
    )
    .await
    .expect("insert catch-all entity");

    // Orthogonal vectors: existing type at dim0, proposal at dim1 → cosine
    // 0.0, well below TYPE_NOVELTY_LOWER_BAND (0.70) → GateOutcome::Pass
    // with existing_name=Some + desc_cosine=Some(0.0) — a REAL comparison.
    let mut vectors = std::collections::HashMap::new();
    vectors.insert(
        "A meteorological occurrence such as a storm or heatwave.".to_string(),
        vec![1.0f32, 0.0, 0.0, 0.0],
    );
    vectors.insert(
        "A legal instrument transferring ownership of real property.".to_string(),
        vec![0.0f32, 1.0, 0.0, 0.0],
    );
    let embedder = MockEmbeddingProvider { vectors };

    let llm = ScriptedSequenceProvider::new(vec![
        r#"{"proposals":[{"name":"PropertyDeed","description":"A legal instrument transferring ownership of real property.","justification":"catch-all evidence"}]}"#,
    ]);

    let result = discover_types(
        &llm,
        DiscoverTypesParams {
            conn: &conn,
            group_id: "g_audit",
            embedder: Some(&embedder),
            max_proposals: 3,
            model_id: "test-model",
            llm_verify_band: false,
            evidence_retype_by_similarity: false,
        },
    )
    .await
    .expect("discover_types must succeed");

    assert_eq!(
        result.types_accepted.len(),
        1,
        "orthogonal, distinct proposal must be accepted, rejected={:?}",
        result.types_rejected
    );

    let mut rows = conn
        .query(
            "SELECT cosine, decision, structural_signal FROM identity_verdict_audit \
             WHERE site = 'site2_type_novelty' AND group_id = 'g_audit' \
             AND candidate_a = 'PropertyDeed'",
            (),
        )
        .await
        .expect("query identity_verdict_audit");
    let row = rows.next().await.expect("row read").expect(
        "an accepted Pass-0 proposal must leave an identity_verdict_audit row \
             — this is the RED assertion: pre-fix, discover_types never wrote to this table \
             on the Pass path at all, so this query returns zero rows",
    );
    let cosine: Option<f64> = row.get(0).expect("cosine column");
    let decision: String = row.get(1).expect("decision column");
    let structural_signal: bool = row.get(2).expect("structural_signal column");
    assert!(
        cosine.is_some(),
        "an accepted proposal compared against a real existing type must persist a \
         NON-NULL cosine — the evidence for WHY it was accepted"
    );
    assert_eq!(decision, "accept");
    assert!(
        !structural_signal,
        "\"PropertyDeed\"/\"WeatherEvent\" share no lemma or exact-name overlap"
    );
}
