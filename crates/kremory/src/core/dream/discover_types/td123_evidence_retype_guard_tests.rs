// ─── Cosine-alone evidence-retype guard ──────────────────────────────────────
//
// A 7th cosine-alone-write site (outside the other enumerated sites):
// `retype_evidence_by_similarity` embeds a BARE ENTITY NAME and
// compares it to the newly-accepted TYPE's DESCRIPTION embedding — the same
// degenerate-embedding failure class documented elsewhere (short bare labels
// collapse to near-identical vectors under weak embedders), just cross-domain
// (name vs. description) rather than name-vs-name. This test simulates that
// exact degeneracy: an UNRELATED catch-all entity's name embeds IDENTICALLY
// (cosine = 1.0) to a newly-discovered, semantically-unrelated type's
// description — proving the DEFAULT build does not wrongly retype it.

use super::*;
use crate::core::entity_types::ensure_default_types_seeded;
use crate::core::provider::{
    ChatMessage, ChatResponse, LLMError, MockChatResponse, StructuredOutputFormat, Tool,
};
use crate::core::schema::TemporalGraph;

#[derive(Debug)]
struct ScriptedProposalProvider {
    json: String,
}

#[async_trait::async_trait]
impl ChatProvider for ScriptedProposalProvider {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
        Ok(Box::new(MockChatResponse {
            text: self.json.clone(),
        }))
    }
}

/// Deterministic embedder: pre-registered vector per input text (exact
/// match), zero vector otherwise (mirrors `site2_type_novelty_tests`'s
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

async fn seed_catch_all(conn: &libsql::Connection, group_id: &str, entity_id: &str) {
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
         VALUES (?1, 0, ?2, ?3)",
        libsql::params![entity_id.to_string(), now, group_id.to_string()],
    )
    .await
    .expect("insert catch-all entity");
}

async fn entity_type_id(conn: &libsql::Connection, group_id: &str, entity_id: &str) -> i64 {
    let mut rows = conn
        .query(
            "SELECT entity_type_id FROM entities WHERE id = ?1 AND group_id = ?2",
            libsql::params![entity_id.to_string(), group_id.to_string()],
        )
        .await
        .expect("query entity_type_id");
    let row = rows
        .next()
        .await
        .expect("row read")
        .expect("entity must exist");
    row.get::<i64>(0).expect("entity_type_id column")
}

const DEGENERATE_PROPOSAL_JSON: &str = r#"{"proposals":[{"name":"Recipe","description":"A step-by-step cooking guide with ingredients and cook time.","justification":"catch-all evidence suggests a recipe type."}]}"#;

/// RED (pre-fix): with the OLD unguarded code, `cosine("ibm", recipe_desc)
/// = 1.0 >= EVIDENCE_RETYPE_COSINE (0.75)` retypes "ibm" (an unrelated
/// catch-all entity) to the newly-discovered "Recipe" type — a false
/// retype driven by cosine alone, with zero lexical/LLM corroboration.
///
/// GREEN (post-fix): `DreamOpts::include_evidence_retype_by_similarity`
/// defaults `false`, so the cosine-only retype path is skipped entirely —
/// "ibm" stays catch-all (`entity_type_id = 0`), left for Pass 2
/// `reclassify`'s LLM+confidence gate to handle safely.
#[tokio::test]
async fn default_flag_off_does_not_retype_on_degenerate_cosine_collision() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let conn = graph.conn.clone();
    ensure_default_types_seeded(&conn, "g1")
        .await
        .expect("seed defaults 0..=9");
    seed_catch_all(&conn, "g1", "ibm").await;

    // Degenerate collision: the catch-all entity's bare name embeds
    // IDENTICALLY to the new type's description (cosine = 1.0), simulating
    // an anisotropic embedder that does not discriminate unrelated bare
    // labels (`cos(Person, Date) = 1.0000`, cross-domain).
    let collision_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
    let mut vectors = std::collections::HashMap::new();
    vectors.insert("ibm".to_string(), collision_vec.clone());
    vectors.insert(
        "A step-by-step cooking guide with ingredients and cook time.".to_string(),
        collision_vec,
    );
    let embedder = MockEmbeddingProvider { vectors };

    let llm = ScriptedProposalProvider {
        json: DEGENERATE_PROPOSAL_JSON.to_string(),
    };

    let result = discover_types(
        &llm,
        DiscoverTypesParams {
            conn: &conn,
            group_id: "g1",
            embedder: Some(&embedder),
            max_proposals: 3,
            model_id: "test-model",
            llm_verify_band: false,
            evidence_retype_by_similarity: false, // DEFAULT
        },
    )
    .await
    .expect("discover_types must succeed");

    assert_eq!(
        result.types_accepted.len(),
        1,
        "the Recipe type itself must still be accepted (distinct from \
         existing defaults) — only the RETYPE decision is guarded, got {:?}",
        result.types_rejected
    );
    assert_eq!(
        result.entities_retyped, 0,
        "flag OFF (default): the cosine-only retype path must be entirely \
         skipped — a degenerate collision must not retype 'ibm' into 'Recipe'"
    );
    assert_eq!(
        entity_type_id(&conn, "g1", "ibm").await,
        0,
        "'ibm' must remain catch-all (entity_type_id=0) — a bare-name-vs-\
         type-description cosine collision is not a valid retype signal \
         without corroboration"
    );
}

/// Regression: explicitly opting IN to the pre-existing behaviour still
/// retypes on the same degenerate collision — proves the flag genuinely
/// gates the code path (not a permanently-dead branch) and preserves the
/// escape hatch for a caller who has independently validated the threshold.
#[tokio::test]
async fn flag_on_preserves_pre_td123_cosine_retype_behaviour() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    let conn = graph.conn.clone();
    ensure_default_types_seeded(&conn, "g2")
        .await
        .expect("seed defaults 0..=9");
    seed_catch_all(&conn, "g2", "ibm").await;

    let collision_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
    let mut vectors = std::collections::HashMap::new();
    vectors.insert("ibm".to_string(), collision_vec.clone());
    vectors.insert(
        "A step-by-step cooking guide with ingredients and cook time.".to_string(),
        collision_vec,
    );
    let embedder = MockEmbeddingProvider { vectors };

    let llm = ScriptedProposalProvider {
        json: DEGENERATE_PROPOSAL_JSON.to_string(),
    };

    let result = discover_types(
        &llm,
        DiscoverTypesParams {
            conn: &conn,
            group_id: "g2",
            embedder: Some(&embedder),
            max_proposals: 3,
            model_id: "test-model",
            llm_verify_band: false,
            evidence_retype_by_similarity: true, // explicit opt-in
        },
    )
    .await
    .expect("discover_types must succeed");

    assert_eq!(
        result.entities_retyped, 1,
        "flag ON: opt-in preserves the pre-existing cosine-only retype \
         behaviour on the same degenerate collision"
    );
    assert_ne!(
        entity_type_id(&conn, "g2", "ibm").await,
        0,
        "flag ON: 'ibm' is retyped away from catch-all, matching the pre-existing \
         behaviour exactly"
    );
}
