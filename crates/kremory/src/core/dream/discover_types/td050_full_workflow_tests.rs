// ─── Deterministic full-workflow test ─────────────────────────────────────────
//
// The regression tests above drive `accept_proposal` (the persistence STEP) in
// isolation. The only test exercising the FULL chain (load catch-alls → cluster
// → LLM proposal → shape-validate → accept → retype evidence) was the `#[ignore]`d
// real-LLM smoke (`tests/phase_d_pass_0.rs`), which is stochastic + shape-only
// ("types_discovered is stochastic — type-shape check only"). So the SEMANTIC
// correctness of discovery — does it grow the table by the proposed type AND
// retype the catch-all evidence with `entity_type_source='DreamPass0'`? — was
// unasserted in the default `cargo test` gate.
//
// This test closes it deterministically: a scripted `ChatProvider` returns a
// KNOWN proposal batch and `embedder = None` takes the degraded-mode path
// (anti-redundancy gate skipped → no embedding-similarity nondeterminism), so
// the OUTCOME is fully determined and asserted. No live LLM; runs in CI.

use super::*;
use crate::core::entity_types::ensure_default_types_seeded;
use crate::core::provider::{
    ChatMessage, ChatResponse, LLMError, MockChatResponse, StructuredOutputFormat, Tool,
};
use crate::core::schema::TemporalGraph;

/// Scripted `ChatProvider` that returns one fixed discovery-proposal batch,
/// ignoring the prompt entirely (mirrors `tests/helpers/scripted_llm.rs`).
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

/// Full workflow: discover_types proposes a KNOWN type, grows `entity_types`
/// above the seeded range, and retypes ALL catch-all evidence in-place with
/// `entity_type_source='DreamPass0'`. Asserts the OUTCOME, not the shape.
#[tokio::test]
async fn discover_types_grows_table_and_retypes_evidence_deterministically() {
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open_in_memory");
    let conn = graph.conn.clone();
    ensure_default_types_seeded(&conn, "legal")
        .await
        .expect("seed defaults 0..=9");

    // Three out-of-vocab entities parked as catch-all (entity_type_id = 0) —
    // the residue Pass-0 discovery operates on. Minimal-column INSERT per the
    // precedent at schema.rs (id, entity_type_id, recorded_at, group_id).
    let now = Utc::now().to_rfc3339();
    for id in ["vanguard therapeutics", "acme capital", "nexus ventures"] {
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
             VALUES (?1, 0, ?2, ?3)",
            libsql::params![id.to_string(), now.clone(), "legal".to_string()],
        )
        .await
        .expect("insert catch-all entity");
    }

    let llm = ScriptedProposalProvider {
        json: r#"{"proposals":[{"name":"Company","description":"A business organisation, firm, or investment fund.","justification":"Vanguard Therapeutics, Acme Capital and Nexus Ventures are all companies."}]}"#
            .to_string(),
    };

    let result = discover_types(
        &llm,
        DiscoverTypesParams {
            conn: &conn,
            group_id: "legal",
            embedder: None,
            max_proposals: 3,
            model_id: "test-model",
            llm_verify_band: false,
            evidence_retype_by_similarity: false,
        },
    )
    .await
    .expect("discover_types must succeed");

    // ── Outcome of the discovery result (not shape) ───────────────────────
    assert_eq!(result.types_proposed.len(), 1, "exactly one type proposed");
    assert_eq!(result.types_accepted.len(), 1, "exactly one type accepted");
    assert_eq!(
        result.types_accepted[0].name, "Company",
        "the accepted type is the one the scripted LLM proposed"
    );
    assert!(
        result.types_rejected.is_empty(),
        "'Company' is a valid name — must not be rejected, got {:?}",
        result.types_rejected
    );
    assert_eq!(
        result.entities_retyped, 3,
        "all 3 catch-all entities retyped in degraded mode"
    );

    // ── entity_types table grew by the KNOWN type, id above seeded range ──
    let new_id: i64 = {
        let mut rows = conn
            .query(
                "SELECT id FROM entity_types WHERE group_id = 'legal' AND name = 'Company'",
                (),
            )
            .await
            .expect("query discovered type");
        rows.next()
            .await
            .expect("row iter")
            .expect("'Company' row must exist — discovery must have persisted it")
            .get::<i64>(0)
            .expect("id column")
    };
    assert!(
        new_id > 9,
        "discovered type id {new_id} must be above the seeded 0..=9 range"
    );

    // ── every catch-all entity retyped to the new id with DreamPass0 ──────
    let mut rows = conn
        .query(
            "SELECT entity_type_id, entity_type_source FROM entities \
             WHERE group_id = 'legal'",
            (),
        )
        .await
        .expect("query retyped entities");
    let mut checked = 0usize;
    while let Some(r) = rows.next().await.expect("row iter") {
        let tid: i64 = r.get(0).expect("entity_type_id");
        let src: String = r.get(1).expect("entity_type_source");
        assert_eq!(
            tid, new_id,
            "every catch-all entity must be retyped to the discovered type id"
        );
        assert_eq!(
            src, "DreamPass0",
            "retype provenance must be 'DreamPass0'"
        );
        checked += 1;
    }
    assert_eq!(
        checked, 3,
        "all 3 entities present and retyped — none left at id=0"
    );
}

/// Fail-loud: when discovery engages on a non-empty catch-all bucket
/// but accepts ZERO types (here the scripted proposal is a `"..."` placeholder
/// the shape validator rejects — the exact gemma4-e2b real-world failure), the
/// result must carry a loud, consumer-visible warning, NOT silently look like
/// "nothing to discover". Evidence must NOT be retyped.
#[tokio::test]
async fn discover_types_warns_loud_when_all_proposals_rejected() {
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open_in_memory");
    let conn = graph.conn.clone();
    ensure_default_types_seeded(&conn, "med")
        .await
        .expect("seed defaults 0..=9");

    let now = Utc::now().to_rfc3339();
    for id in ["aspirin", "ibuprofen", "paracetamol"] {
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
             VALUES (?1, 0, ?2, ?3)",
            libsql::params![id.to_string(), now.clone(), "med".to_string()],
        )
        .await
        .expect("insert catch-all entity");
    }

    // Scripted placeholder name — rejected by the validator (ellipsis_placeholder).
    let llm = ScriptedProposalProvider {
        json: r#"{"proposals":[{"name":"...","description":"x","justification":"y"}]}"#
            .to_string(),
    };

    let result = discover_types(
        &llm,
        DiscoverTypesParams {
            conn: &conn,
            group_id: "med",
            embedder: None,
            max_proposals: 3,
            model_id: "test-model",
            llm_verify_band: false,
            evidence_retype_by_similarity: false,
        },
    )
    .await
    .expect("discover_types must succeed (zero-discovery is non-fatal)");

    assert_eq!(result.types_proposed.len(), 1, "one proposal seen");
    assert!(
        result.types_accepted.is_empty(),
        "the '...' placeholder must be rejected → zero accepted"
    );
    assert_eq!(result.types_rejected.len(), 1, "one rejection");
    assert_eq!(
        result.entities_retyped, 0,
        "nothing accepted → nothing retyped"
    );

    // The load-bearing assertion: zero-discovery is LOUD, not silent.
    assert!(
        result.warnings.iter().any(|w| w.contains("ZERO accepted")),
        "a non-empty catch-all bucket yielding zero accepted types must emit a \
         consumer-visible warning; warnings={:?}",
        result.warnings
    );

    // Evidence untouched — still catch-all.
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE group_id = 'med' AND entity_type_id = 0",
            (),
        )
        .await
        .expect("count");
    let still_catch_all: i64 = rows
        .next()
        .await
        .expect("row")
        .expect("count row")
        .get::<i64>(0)
        .expect("count col");
    assert_eq!(
        still_catch_all, 3,
        "all 3 entities remain catch-all when discovery accepts nothing"
    );
}
