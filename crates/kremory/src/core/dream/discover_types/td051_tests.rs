// ─── Regression tests: composite-PK INSERT id omission ───────────────────────
//
// `accept_proposal` is the Pass-0 persistence step. It was buried below
// the LLM call + clustering + anti-redundancy gate, so the only test exercising
// it was the `#[ignore]`d real-LLM smoke (`tests/phase_d_pass_0.rs`) — which hid
// a bug where INSERT omitted `id` on a composite-PK table → runtime NOT NULL crash.
// These tests drive the REAL `accept_proposal` deterministically (no LLM, no
// embedder; empty `catch_alls` makes retype a no-op) so the persistence/id
// allocation is asserted in the default `cargo test` gate.

use super::*;
use crate::core::entity_types::ensure_default_types_seeded;
use crate::core::schema::TemporalGraph;

async fn accepted_type_id(conn: &libsql::Connection, group_id: &str, name: &str) -> i64 {
    let mut rows = conn
        .query(
            "SELECT id FROM entity_types WHERE group_id = ?1 AND name = ?2",
            libsql::params![group_id, name],
        )
        .await
        .expect("select discovered type");
    rows.next()
        .await
        .expect("row iter")
        .expect("discovered-type row must exist — INSERT must not have crashed")
        .get::<i64>(0)
        .expect("id column")
}

/// First discovered type allocates `id = 10` — above the seeded range (0..=9).
/// Before this fix, this panicked with `NOT NULL constraint failed`.
#[tokio::test]
async fn accept_proposal_allocates_id_above_seed_range() {
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open_in_memory");
    let conn = graph.conn.clone();
    ensure_default_types_seeded(&conn, "g1")
        .await
        .expect("seed defaults 0..=9");

    let proposal = TypeProposal {
        name: "Vehicle".to_string(),
        description: "A car, truck, or other conveyance.".to_string(),
        justification: "Several catch-all entities were vehicles.".to_string(),
    };
    let mut result = DiscoveryResult::default();

    accept_proposal(AcceptProposalParams {
        conn: &conn,
        group_id: "g1",
        model_str: "test-model",
        proposal: &proposal,
        catch_alls: &[],
        desc_emb_and_embedder: None,
        evidence_retype_by_similarity: false,
        result: &mut result,
    })
    .await
    .expect("accept_proposal must persist the discovered type");

    assert_eq!(
        accepted_type_id(&conn, "g1", "Vehicle").await,
        10,
        "first discovered type allocates id=10 (above seeded 0..=9)"
    );
    assert_eq!(result.types_accepted.len(), 1, "one type accepted");
    assert_eq!(result.types_accepted[0].name, "Vehicle");
}

/// Successive discoveries climb `MAX(id)+1` (10, 11) without colliding with
/// the seeded range or the id=0 catch-all.
#[tokio::test]
async fn accept_proposal_increments_id_across_discoveries() {
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open_in_memory");
    let conn = graph.conn.clone();
    ensure_default_types_seeded(&conn, "g1")
        .await
        .expect("seed");

    for (name, expected_id) in [("Vehicle", 10i64), ("Statute", 11i64)] {
        let proposal = TypeProposal {
            name: name.to_string(),
            description: format!("description for {name}"),
            justification: "j".to_string(),
        };
        let mut result = DiscoveryResult::default();
        accept_proposal(AcceptProposalParams {
            conn: &conn,
            group_id: "g1",
            model_str: "m",
            proposal: &proposal,
            catch_alls: &[],
            desc_emb_and_embedder: None,
            evidence_retype_by_similarity: false,
            result: &mut result,
        })
        .await
        .expect("accept_proposal persists");
        assert_eq!(
            accepted_type_id(&conn, "g1", name).await,
            expected_id,
            "{name} must allocate id={expected_id}"
        );
    }

    // id=0 catch-all is untouched by discovery.
    assert_eq!(accepted_type_id(&conn, "g1", "Entity").await, 0);
}
