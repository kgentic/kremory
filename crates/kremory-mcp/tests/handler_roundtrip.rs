//! In-process handler round-trip tests for the floor-5 tool surface.
//!
//! Each test constructs a real `kremory::Memory` on an in-memory libSQL db
//! wired with `MockChatProvider::null()` + `NullEmbeddingProvider` (no live
//! model needed), builds a `KremoryMcpServer` over it, and calls the tool
//! handlers directly with `Parameters(...)`.
//!
//! ## On the DOD-001 recall proof + the enrichment seam
//!
//! The spec's DOD-001 asks that a `remember` with pinned `structured_facts` +
//! `skip_extraction=true` be provable through a `recall` that returns ≥1
//! result. Investigation against the substrate (2026-07-14) showed this is
//! NOT achievable with mock providers *alone*: kremory's recall entity arm
//! surfaces an entity only via (a) the entity FTS index or (b) the entity
//! vector — and BOTH are populated by the LLM enrichment / verify stage,
//! which `skip_extraction` deliberately skips (and which `MockChatProvider`
//! cannot perform anyway). A `skip_extraction` pinned entity lands with an
//! EMPTY FTS label and no embedding, so recall-by-name returns 0. kremory's
//! own `with_facts_integration.rs` documents the same reality — it verifies
//! pins via a substrate COUNTER, not via recall.
//!
//! So the DOD-001 read-path proof here [`recall_returns_pinned_entity_after_
//! enrichment`] writes via the `kremory_remember` tool, then simulates the
//! ONE thing the mock LLM can't do — populate the FTS-indexed entity name the
//! verify stage would write — using kremory's sanctioned
//! `temporal_graph_for_test` seam, and then proves the `kremory_recall` tool
//! surfaces that entity end-to-end. The enrichment step is a clearly-scoped
//! test seam, not production behaviour.

use std::sync::Arc;

use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
use kremory::{ChatProvider, DynEmbeddingProvider, Memory, Namespace};
use kremory_mcp::params::{
    DreamParams, ListMutationsParams, RecallFormat, RecallParams, RecallTemplateWire,
    RememberParams, SourceKindWire, StructuredFactWire, UndoParams,
};
use kremory_mcp::KremoryMcpServer;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::ErrorCode;

/// Build an in-memory Memory (mock LLM + null embedder). Returned as an `Arc`
/// so a test can both hand it to the server AND keep a handle to drive the
/// `temporal_graph_for_test` enrichment seam.
async fn mock_memory() -> Arc<Memory> {
    let llm: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
    let mem = Memory::open(":memory:")
        .with_llm(llm)
        .with_embedder(embedder)
        .await
        .expect("in-memory Memory must build");
    Arc::new(mem)
}

async fn mock_server() -> KremoryMcpServer {
    KremoryMcpServer::new(mock_memory().await)
}

/// A mode-(c) remember: pin a structured fact + skip LLM extraction. No live
/// model needed — the pinned fact + its subject entity are written directly.
fn pinned_remember_params(namespace: &str, subject: &str) -> RememberParams {
    RememberParams {
        namespace: namespace.to_string(),
        thread: None,
        content: format!("{subject} leads the design team"),
        source_kind: Some(SourceKindWire::Note),
        source_id: Some("doc-1".into()),
        published_at: None,
        structured_facts: vec![StructuredFactWire {
            subject: subject.to_string(),
            predicate: "leads".into(),
            object: "design".into(),
            valid_at: None,
            invalid_at: None,
        }],
        skip_extraction: true,
    }
}

/// Simulate the LLM enrichment / verify stage for a single entity: write the
/// entity NAME into the FTS-indexed `properties` column (both the `entities`
/// row and its `entities_fts` shadow row), which is what makes an entity
/// findable by name via recall. `MockChatProvider` cannot do this; a live
/// model does it as part of extraction. Uses kremory's `test-utils`
/// `temporal_graph_for_test` seam.
async fn simulate_enrichment(mem: &Memory, entity_id: &str) {
    let tg = mem
        .temporal_graph_for_test()
        .expect("Memory built via the builder path exposes a TemporalGraph");
    let props = format!("{{\"name\":\"{entity_id}\",\"stub\":false}}");
    tg.conn
        .execute(
            "UPDATE entities SET properties = ?2 WHERE id = ?1",
            (entity_id.to_string(), props.clone()),
        )
        .await
        .expect("seed entity properties");
    tg.conn
        .execute(
            "UPDATE entities_fts SET properties = ?2 WHERE entity_id = ?1",
            (entity_id.to_string(), props),
        )
        .await
        .expect("seed entities_fts properties");
}

// ─── remember (mode-c pinned path) ───────────────────────────────────────

#[tokio::test]
async fn remember_pinned_fact_returns_commit_output() {
    let server = mock_server().await;
    let result = server
        .kremory_remember(Parameters(pinned_remember_params("ns-remember", "alice")))
        .await
        .expect("mode-c remember must succeed");

    assert_eq!(result.is_error, Some(false));
    let structured = result
        .structured_content
        .expect("remember returns structured payload");
    assert!(
        structured["episode_entity_id"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "episode_entity_id must be a non-empty string: {structured}"
    );
    assert!(
        structured["committed_at"].as_str().is_some(),
        "committed_at must be an ISO-8601 string: {structured}"
    );
}

// ─── recall shape / bridge proof (mock tier, no enrichment) ──────────────

/// Without enrichment a mock-tier recall may legitimately return 0 results
/// (see module docs). This test proves the BRIDGE: the `kremory_recall` tool
/// returns a well-formed structured payload (`count` + `results` array, with
/// `count == results.len()`) and never errors — regardless of hit count.
#[tokio::test]
async fn recall_structured_returns_wellformed_payload() {
    let server = mock_server().await;
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-shape", "alice")))
        .await
        .expect("remember must succeed");

    let recall = RecallParams {
        namespace: "ns-shape".into(),
        thread: None,
        query: "alice".into(),
        k: Some(10),
        as_of: None,
        format: RecallFormat::Structured,
        template: RecallTemplateWire::default(),
    };
    let result = server
        .kremory_recall(Parameters(recall))
        .await
        .expect("recall must not error");

    let structured = result
        .structured_content
        .expect("structured recall returns structured payload");
    let count = structured["count"].as_u64().expect("count present");
    let results = structured["results"].as_array().expect("results array");
    assert_eq!(
        count as usize,
        results.len(),
        "count field must equal results array length: {structured}"
    );
}

// ─── DOD-001: mode-c remember → (simulated enrichment) → recall ≥ 1 ───────

#[tokio::test]
async fn recall_returns_pinned_entity_after_enrichment() {
    let mem = mock_memory().await;
    let server = KremoryMcpServer::new(mem.clone());

    // Distinctive token so the entity FTS arm can find it unambiguously.
    let subject = "Zephyrine";
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-dod001", subject)))
        .await
        .expect("mode-c remember must succeed");

    // The pinned subject entity is created with id == subject slug. Simulate
    // the enrichment the mock LLM cannot perform (populate the FTS-indexed
    // name), then prove the recall tool surfaces it.
    simulate_enrichment(&mem, subject).await;

    let recall = RecallParams {
        namespace: "ns-dod001".into(),
        thread: None,
        query: subject.to_string(),
        k: Some(10),
        as_of: None,
        format: RecallFormat::Structured,
        template: RecallTemplateWire::default(),
    };
    let result = server
        .kremory_recall(Parameters(recall))
        .await
        .expect("recall must succeed");

    let structured = result
        .structured_content
        .expect("structured recall returns structured payload");
    let count = structured["count"].as_u64().expect("count present");
    let results = structured["results"].as_array().expect("results array");
    assert!(
        count >= 1,
        "DOD-001: recall after remember + enrichment must return >=1 result, got count={count}: {structured}"
    );
    let names: Vec<&str> = results
        .iter()
        .filter_map(|r| r["entity_name"].as_str())
        .collect();
    assert!(
        names.iter().any(|n| n.contains(subject)),
        "recall results must contain the remembered entity {subject:?}, got names={names:?}"
    );
}

// ─── ADR-074 review H2: facts must survive the MCP wire layer ────────────

/// Regression guard for review finding H2: the only prior end-to-end proof
/// that `facts` survives projection lived in `kremory`'s own
/// `with_facts_integration.rs` (`td116_recall_returns_connected_facts_...`);
/// this crate's one same-named-looking test
/// (`recall_structured_returns_wellformed_payload`) builds no facts fixture
/// at all, so nothing here ever asserted the MCP wire mapping.
///
/// A mode-(c) pinned fact does NOT need the `simulate_enrichment` seam this
/// module's docs describe for free-text/LLM-extracted entities: TD-113
/// stamps the FTS-indexed name (`properties["name"]`) and a vector embedding
/// at PIN time (`ingest_with.rs::make_pinned_entity_recallable`), so the
/// subject is recall-findable — and its fact attaches — from the pin alone.
/// Verified empirically before writing this assertion (a probe recall
/// without `simulate_enrichment` already returned a non-empty `facts` array).
#[tokio::test]
async fn recall_structured_surfaces_pinned_fact_with_every_wire_field() {
    let server = mock_server().await;

    let subject = "Wilhelmina";
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-facts-h2", subject)))
        .await
        .expect("mode-c remember must succeed");

    let recall = RecallParams {
        namespace: "ns-facts-h2".into(),
        thread: None,
        query: subject.to_string(),
        k: Some(10),
        as_of: None,
        format: RecallFormat::Structured,
        template: RecallTemplateWire::default(),
    };
    let result = server
        .kremory_recall(Parameters(recall))
        .await
        .expect("recall must succeed");

    let structured = result
        .structured_content
        .expect("structured recall returns structured payload");
    let results = structured["results"]
        .as_array()
        .expect("results array present");
    let subject_result = results
        .iter()
        .find(|r| r["entity_name"].as_str() == Some(subject))
        .unwrap_or_else(|| panic!("recall must surface {subject:?}, got: {structured}"));

    let facts = subject_result["facts"]
        .as_array()
        .expect("facts field present and typed as an array");
    assert!(
        !facts.is_empty(),
        "H2: {subject}'s pinned 'leads' fact must survive the MCP wire mapping \
         (RetrievedContext → RetrievedContextWire), got empty facts: {structured}"
    );

    let fact = facts
        .iter()
        .find(|f| f["predicate"].as_str() == Some("leads"))
        .unwrap_or_else(|| panic!("the pinned 'leads' fact must be present, got: {facts:?}"));

    // Every `RetrievedFactWire` field must round-trip with the pinned triple's
    // real values — not just be present-and-typed.
    assert_eq!(fact["fact"].as_str(), Some("Wilhelmina leads design"));
    assert_eq!(fact["subject"].as_str(), Some("Wilhelmina"));
    assert_eq!(fact["predicate"].as_str(), Some("leads"));
    assert_eq!(fact["object"].as_str(), Some("design"));
    assert_eq!(fact["object_is_entity"].as_bool(), Some(false));
    assert!(
        fact["valid_at"].as_str().is_some_and(|s| !s.is_empty()),
        "valid_at must be a non-empty RFC-3339 string: {fact:?}"
    );
    assert!(
        fact["invalid_at"].is_null(),
        "an open-ended pinned fact must have invalid_at=null: {fact:?}"
    );
    assert!(
        fact["recorded_at"].as_str().is_some_and(|s| !s.is_empty()),
        "recorded_at must be a non-empty RFC-3339 string: {fact:?}"
    );
    assert!(
        fact["expired_at"].is_null(),
        "a fresh pinned fact must have expired_at=null: {fact:?}"
    );
    assert_eq!(
        fact["confidence"].as_f64(),
        Some(1.0),
        "caller-pinned facts default to confidence=1.0: {fact:?}"
    );
    let source_episode_ids = fact["source_episode_ids"]
        .as_array()
        .expect("source_episode_ids present and typed as an array");
    assert!(
        !source_episode_ids.is_empty(),
        "source_episode_ids must attribute the fact to its episode: {fact:?}"
    );
    assert!(
        fact["score"].as_f64().is_some(),
        "score must be present and numeric: {fact:?}"
    );
}

// ─── recall (text format) ─────────────────────────────────────────────────

#[tokio::test]
async fn recall_text_renders_entity_after_enrichment() {
    let mem = mock_memory().await;
    let server = KremoryMcpServer::new(mem.clone());

    let subject = "Bartholomew";
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-text", subject)))
        .await
        .expect("remember must succeed");
    simulate_enrichment(&mem, subject).await;

    let recall = RecallParams {
        namespace: "ns-text".into(),
        thread: None,
        query: subject.to_string(),
        k: Some(10),
        as_of: None,
        format: RecallFormat::Text,
        // Entities template renders name + summary regardless of source_refs
        // (temporal_facts needs episodic edges the pinned entity may lack).
        template: RecallTemplateWire::Entities,
    };
    let result = server
        .kremory_recall(Parameters(recall))
        .await
        .expect("recall text must succeed");

    let structured = result
        .structured_content
        .expect("text recall returns structured payload with a `block` field");
    let block = structured["block"].as_str().expect("block is a string");
    assert!(
        block.contains(subject),
        "entities template block must mention the remembered entity: {block:?}"
    );
}

// ─── dream ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn dream_returns_summary_outcome() {
    let server = mock_server().await;
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-dream", "Cornelius")))
        .await
        .expect("remember must succeed");

    let dream = DreamParams {
        namespace: "ns-dream".into(),
        thread: None,
        batch_id: None,
    };
    let result = server
        .kremory_dream(Parameters(dream))
        .await
        .expect("dream must succeed with the mock LLM");

    assert_eq!(result.is_error, Some(false));
    let structured = result
        .structured_content
        .expect("dream returns structured payload");
    // Honest-zero fields are fine (mock LLM); the contract is that the real
    // DreamSummary fields are present + typed, not that they are non-zero.
    assert!(
        structured["duration_ms"].is_u64(),
        "dream output must carry duration_ms: {structured}"
    );
    assert!(
        structured["consolidation_ops_ran"].is_object(),
        "dream output must carry the consolidation_ops_ran object: {structured}"
    );
    assert!(
        structured["warnings"].is_array(),
        "dream output must carry a warnings array: {structured}"
    );
    // G2 — dream must surface WHAT it discovered, not just a count. The mock LLM
    // may discover zero types, so this asserts the array is present + typed (the
    // name/description/justification detail is proven by the conversions.rs unit
    // test `dream_output_carries_type_proposal_detail`).
    assert!(
        structured["types_discovered"].is_array(),
        "dream output must carry a types_discovered array: {structured}"
    );
}

// ─── list_mutations / undo (G3 — the reversible-mutations gap closure) ────

/// `kremory_list_mutations` after a `dream()` run must return a well-formed
/// typed array (`count == results.len()` is N/A here — the tool returns the
/// array directly, not a `{results, count}` envelope). The mock LLM may
/// discover/merge nothing, so the load-bearing assertion is shape, not count
/// — mirrors `recall_structured_returns_wellformed_payload`'s discipline.
#[tokio::test]
async fn list_mutations_after_dream_returns_wellformed_array() {
    let server = mock_server().await;
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-list", "Percival")))
        .await
        .expect("remember must succeed");
    server
        .kremory_dream(Parameters(DreamParams {
            namespace: "ns-list".into(),
            thread: None,
            batch_id: None,
        }))
        .await
        .expect("dream must succeed with the mock LLM");

    let list_params = ListMutationsParams {
        namespace: "ns-list".into(),
        thread: None,
        entity_id: None,
        kind: None,
        since: None,
        include_undone: Some(true),
    };
    let result = server
        .kremory_list_mutations(Parameters(list_params))
        .await
        .expect("list_mutations must not error");

    assert_eq!(result.is_error, Some(false));
    let payload = result
        .structured_content
        .expect("list_mutations returns structured payload");
    assert!(
        payload.is_array(),
        "kremory_list_mutations must return a typed array: {payload}"
    );
}

/// `MutationRecord` / `UndoOutcome` are `#[non_exhaustive]` facade types
/// kremory-mcp cannot struct-literal-construct in a unit test (see the
/// conversions.rs test module docs on this file's counterpart). This test
/// proves `From<MutationRecord> for MutationRecordWire` and
/// `TryFrom<UndoOutcome> for UndoOutcomeWire` preserve every field through a
/// REAL facade round-trip: `edit_entity(...).rename(...)` needs no LLM and
/// logs a genuine `entity_edit` mutation, which `kremory_list_mutations`
/// (SEE) then `kremory_undo` (FIX) must surface + reverse with every count
/// intact — the exact parity-drop class ADR-074 / G2 fixed elsewhere in this
/// crate.
#[tokio::test]
async fn list_mutations_and_undo_roundtrip_entity_edit() {
    let mem = mock_memory().await;
    let server = KremoryMcpServer::new(mem.clone());

    let subject = "Reginald";
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-undo", subject)))
        .await
        .expect("mode-c remember must succeed");

    // Drive a genuine `entity_edit` mutation directly via the facade —
    // `edit_entity` needs no LLM (unlike merges, which only arise from
    // dream()'s LLM-driven passes).
    let ns = Namespace::new("ns-undo");
    let edit = mem
        .edit_entity(subject)
        .in_namespace(ns)
        .rename("reginald-ii")
        .execute()
        .await
        .expect("edit_entity must succeed");
    assert!(!edit.already_undone);
    assert!(edit.rekeyed);

    // SEE: kremory_list_mutations surfaces the logged entity_edit — proves
    // every MutationRecordWire field survives the projection.
    let list_params = ListMutationsParams {
        namespace: "ns-undo".into(),
        thread: None,
        entity_id: None,
        kind: Some("entity_edit".into()),
        since: None,
        include_undone: None,
    };
    let list_result = server
        .kremory_list_mutations(Parameters(list_params))
        .await
        .expect("list_mutations must not error");
    let records = list_result
        .structured_content
        .expect("list_mutations returns structured payload");
    let records = records.as_array().expect("results must be an array");
    assert_eq!(
        records.len(),
        1,
        "expected exactly one live entity_edit mutation: {records:?}"
    );
    let record = &records[0];
    assert_eq!(record["mutation_id"], edit.mutation_id);
    assert_eq!(record["kind"], "entity_edit");
    assert_eq!(record["undone"], false);
    assert_eq!(record["group_id"], "ns-undo");
    assert!(
        record["created_at"].as_str().is_some_and(|s| !s.is_empty()),
        "created_at must be a non-empty RFC3339 string: {record}"
    );
    let affected = record["affected_entities"]
        .as_array()
        .expect("affected_entities must be an array");
    assert!(
        affected
            .iter()
            .any(|v| v.as_str() == Some("reginald-ii") || v.as_str() == Some(subject)),
        "affected_entities must name the edited entity: {affected:?}"
    );
    assert!(
        record["summary"].as_str().is_some_and(|s| !s.is_empty()),
        "summary must be a non-empty human-readable string: {record}"
    );

    // FIX: kremory_undo dispatches to undo_entity_edit and preserves every
    // per-kind count on the wire (the UndoOutcomeWire::EditEntity variant).
    let undo_params = UndoParams {
        namespace: "ns-undo".into(),
        thread: None,
        mutation_id: edit.mutation_id,
    };
    let undo_result = server
        .kremory_undo(Parameters(undo_params))
        .await
        .expect("undo must succeed");
    assert_eq!(undo_result.is_error, Some(false));
    let outcome = undo_result
        .structured_content
        .expect("undo returns structured payload");
    assert_eq!(outcome["reversed_kind"], "edit_entity");
    assert_eq!(
        outcome["entity_id"], subject,
        "undo must restore the ORIGINAL entity id: {outcome}"
    );
    assert_eq!(outcome["rekeyed"], true);
    assert_eq!(outcome["retyped"], false);
    assert_eq!(outcome["mutation_id"], edit.mutation_id);
    assert_eq!(outcome["already_undone"], false);
    assert!(
        outcome["facts_repointed"].is_u64(),
        "facts_repointed count must survive to the wire: {outcome}"
    );
    assert!(
        outcome["edges_repointed"].is_u64(),
        "edges_repointed count must survive to the wire: {outcome}"
    );

    // A second undo of the same mutation_id is an idempotent no-op — the
    // `already_undone` count survives too.
    let undo_again = UndoParams {
        namespace: "ns-undo".into(),
        thread: None,
        mutation_id: edit.mutation_id,
    };
    let repeat = server
        .kremory_undo(Parameters(undo_again))
        .await
        .expect("repeat undo must not error");
    let repeat_outcome = repeat
        .structured_content
        .expect("repeat undo returns structured payload");
    assert_eq!(repeat_outcome["already_undone"], true);
}

/// `kremory_undo` on a `mutation_id` naming no `graph_mutation_log` row must
/// fail LOUD (`Error::MutationNotFound`, mapped to `internal_error` — the
/// same facade-error mapping every other tool uses), never silently no-op.
#[tokio::test]
async fn undo_nonexistent_mutation_fails_loud() {
    let server = mock_server().await;
    let params = UndoParams {
        namespace: "ns-undo-missing".into(),
        thread: None,
        mutation_id: 987_654_321,
    };
    let err = server
        .kremory_undo(Parameters(params))
        .await
        .expect_err("undo of a nonexistent mutation_id must error, never silently no-op");
    assert_eq!(
        err.code,
        ErrorCode::INTERNAL_ERROR,
        "MutationNotFound is a facade error, mapped to internal_error like every other tool: {err:?}"
    );
    assert!(
        err.message.to_lowercase().contains("mutation"),
        "internal_error must name the mutation-not-found problem: {}",
        err.message
    );
}

#[tokio::test]
async fn list_mutations_rejects_empty_namespace_with_invalid_params() {
    let server = mock_server().await;
    let params = ListMutationsParams {
        namespace: "".into(),
        thread: None,
        entity_id: None,
        kind: None,
        since: None,
        include_undone: None,
    };
    let err = server
        .kremory_list_mutations(Parameters(params))
        .await
        .expect_err("empty namespace must error");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(err.message.to_lowercase().contains("namespace"));
}

#[tokio::test]
async fn list_mutations_rejects_unknown_kind_with_invalid_params() {
    let server = mock_server().await;
    let params = ListMutationsParams {
        namespace: "ns-bad-kind".into(),
        thread: None,
        entity_id: None,
        kind: Some("not_a_real_kind".into()),
        since: None,
        include_undone: None,
    };
    let err = server
        .kremory_list_mutations(Parameters(params))
        .await
        .expect_err("unknown kind string must error");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(
        err.message.to_lowercase().contains("kind"),
        "invalid_params must name the offending kind: {}",
        err.message
    );
}

// ─── concurrency (ASMP-002) ───────────────────────────────────────────────

/// Two overlapping recalls dispatched onto SEPARATE `tokio::spawn` tasks
/// (not `tokio::join!`, which only interleaves both futures cooperatively
/// within a single task and never proves cross-thread simultaneous access).
/// Spawning each recall as its own task on the multi-thread runtime lets the
/// scheduler run them on different OS threads, genuinely exercising
/// concurrent access to the shared `Arc<Memory>` — the real ASMP-002
/// concern (libSQL under concurrent dispatch). Proves `Memory` is
/// `Send + Sync` enough for rmcp's concurrent-handler dispatch (stdio does
/// not serialize calls). If `Memory` were not `Sync` this would not compile.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_overlapping_recalls_run_concurrently() {
    let server = mock_server().await;
    server
        .kremory_remember(Parameters(pinned_remember_params(
            "ns-concurrent",
            "Delphine",
        )))
        .await
        .expect("remember must succeed");

    let recall = |query: &str| RecallParams {
        namespace: "ns-concurrent".into(),
        thread: None,
        query: query.to_string(),
        k: Some(10),
        as_of: None,
        format: RecallFormat::Structured,
        template: RecallTemplateWire::default(),
    };

    let server_a = server.clone();
    let recall_a = recall("Delphine");
    let handle_a = tokio::spawn(async move { server_a.kremory_recall(Parameters(recall_a)).await });

    let server_b = server.clone();
    let recall_b = recall("design");
    let handle_b = tokio::spawn(async move { server_b.kremory_recall(Parameters(recall_b)).await });

    let a = handle_a
        .await
        .expect("first concurrent recall task must not panic");
    let b = handle_b
        .await
        .expect("second concurrent recall task must not panic");
    assert!(a.is_ok(), "first concurrent recall must succeed: {a:?}");
    assert!(b.is_ok(), "second concurrent recall must succeed: {b:?}");
}

// ─── error mapping ────────────────────────────────────────────────────────

#[tokio::test]
async fn remember_rejects_empty_namespace_with_invalid_params() {
    let server = mock_server().await;
    let mut params = pinned_remember_params("x", "alice");
    params.namespace = "".into();
    let err = server
        .kremory_remember(Parameters(params))
        .await
        .expect_err("empty namespace must error");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(
        err.message.to_lowercase().contains("namespace"),
        "invalid_params must name the namespace problem: {}",
        err.message
    );
}

#[tokio::test]
async fn remember_rejects_malformed_published_at_with_invalid_params() {
    let server = mock_server().await;
    let mut params = pinned_remember_params("ns-bad-ts", "alice");
    params.published_at = Some("yesterday at noon".into());
    let err = server
        .kremory_remember(Parameters(params))
        .await
        .expect_err("malformed published_at must error");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(
        err.message.contains("published_at"),
        "invalid_params must name the offending field: {}",
        err.message
    );
}

#[tokio::test]
async fn recall_rejects_empty_namespace_with_invalid_params() {
    let server = mock_server().await;
    let recall = RecallParams {
        namespace: "".into(),
        thread: None,
        query: "x".into(),
        k: None,
        as_of: None,
        format: RecallFormat::Text,
        template: RecallTemplateWire::default(),
    };
    let err = server
        .kremory_recall(Parameters(recall))
        .await
        .expect_err("empty namespace must error");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(err.message.to_lowercase().contains("namespace"));
}

/// `as_of` point-in-time recall is declared-but-unimplemented in the
/// substrate (`kremory::memory::mod.rs` — `opts.as_of.is_some()` returns
/// `Err(Error::Unsupported { feature: "as_of point-in-time recall" })`
/// rather than silently ignoring the filter). `params.rs` documents this as
/// a deliberate fail-loud contract; this test locks that exact behaviour
/// through the `kremory_recall` tool: setting `as_of` must map to MCP
/// `internal_error`, never succeed and never silently drop the filter.
#[tokio::test]
async fn recall_as_of_fails_loud_with_internal_error() {
    let server = mock_server().await;
    server
        .kremory_remember(Parameters(pinned_remember_params("ns-as-of", "alice")))
        .await
        .expect("remember must succeed");

    let recall = RecallParams {
        namespace: "ns-as-of".into(),
        thread: None,
        query: "alice".into(),
        k: None,
        as_of: Some("2026-01-01T00:00:00Z".into()),
        format: RecallFormat::Structured,
        template: RecallTemplateWire::default(),
    };
    let err = server
        .kremory_recall(Parameters(recall))
        .await
        .expect_err("as_of must fail loud, never silently succeed");
    assert_eq!(
        err.code,
        ErrorCode::INTERNAL_ERROR,
        "as_of must map to internal_error (Unsupported), not invalid_params: {err:?}"
    );
    assert!(
        err.message.to_lowercase().contains("as_of"),
        "internal_error must name the unsupported as_of feature: {}",
        err.message
    );
}
