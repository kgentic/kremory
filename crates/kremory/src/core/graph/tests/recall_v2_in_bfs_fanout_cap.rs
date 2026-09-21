use super::super::*;
use crate::core::schema::TemporalGraph;
use chrono::Duration;

// === recall-v2 in-BFS fan-out cap ===

/// The `max_visited` cap must (a) be **byte-identical at hops=1** even for a hub
/// with MORE than `cap` direct neighbours (the cap is checked at the top of the
/// loop, so all hop-1 neighbours are enqueued in the seed's own iteration before
/// it can fire), and (b) **bound the hops=2 traversal**, preventing the
/// hub-explosion that widening the hop count would otherwise reintroduce on the
/// always-on default path.
#[tokio::test]
async fn get_neighbours_at_fan_out_cap_byte_identical_at_hops1_bounds_hops2() {
    // Matches the shipped default `SearchConfig::expansion_fan_out_cap`.
    const CAP: usize = 8;
    const SPOKES: usize = 12; // deliberately > CAP
    const LEAVES_PER_SPOKE: usize = 5;

    let g = TemporalGraph::open_in_memory().await.unwrap();
    let t0 = Utc::now() - Duration::hours(1);

    g.insert_entity(InsertEntityParams {
        id: "hub",
        entity_type_id: 0,
        properties: serde_json::json!({}),
    })
    .await
    .unwrap();
    for s in 0..SPOKES {
        let spoke = format!("spoke{s}");
        g.insert_entity(InsertEntityParams {
            id: &spoke,
            entity_type_id: 0,
            properties: serde_json::json!({}),
        })
        .await
        .unwrap();
        g.insert_fact(FactInsert::new("hub", "connected_to", t0).object_id(&spoke))
            .await
            .unwrap();
        for l in 0..LEAVES_PER_SPOKE {
            let leaf = format!("leaf{s}_{l}");
            g.insert_entity(InsertEntityParams {
                id: &leaf,
                entity_type_id: 0,
                properties: serde_json::json!({}),
            })
            .await
            .unwrap();
            g.insert_fact(FactInsert::new(&spoke, "connected_to", t0).object_id(&leaf))
                .await
                .unwrap();
        }
    }

    fn sorted_ids(sg: &SubGraph) -> Vec<String> {
        let mut v: Vec<String> = sg.entities.iter().map(|e| e.id.clone()).collect();
        v.sort();
        v
    }

    // (a) hops=1 byte-identical: hub + all 12 direct spokes, cap does NOT bite.
    let h1_uncapped = g
        .get_neighbours_at(GetNeighboursAtParams {
            entity_id: "hub",
            hops: 1,
            as_of: None,
            max_visited: None,
        })
        .await
        .unwrap();
    let h1_capped = g
        .get_neighbours_at(GetNeighboursAtParams {
            entity_id: "hub",
            hops: 1,
            as_of: None,
            max_visited: Some(CAP),
        })
        .await
        .unwrap();
    assert_eq!(
        sorted_ids(&h1_capped),
        sorted_ids(&h1_uncapped),
        "hops=1 must be byte-identical: the cap must not drop any direct neighbour"
    );
    assert_eq!(
        h1_uncapped.entities.len(),
        1 + SPOKES,
        "hub + all {SPOKES} direct spokes at hops=1"
    );

    // (b) hops=2 bounded: uncapped explodes to hub+spokes+leaves; capped breaks
    // before expanding the hop-2 frontier, so it stays far smaller.
    let h2_uncapped = g
        .get_neighbours_at(GetNeighboursAtParams {
            entity_id: "hub",
            hops: 2,
            as_of: None,
            max_visited: None,
        })
        .await
        .unwrap();
    let h2_capped = g
        .get_neighbours_at(GetNeighboursAtParams {
            entity_id: "hub",
            hops: 2,
            as_of: None,
            max_visited: Some(CAP),
        })
        .await
        .unwrap();
    assert_eq!(
        h2_uncapped.entities.len(),
        1 + SPOKES + SPOKES * LEAVES_PER_SPOKE,
        "uncapped hops=2 visits the full 2-hop closure (explosion)"
    );
    assert!(
        h2_capped.entities.len() < h2_uncapped.entities.len(),
        "capped hops=2 ({}) must be strictly smaller than uncapped ({}) — the cap \
         bounds hub-explosion (spec R1)",
        h2_capped.entities.len(),
        h2_uncapped.entities.len()
    );
}

