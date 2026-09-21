use super::super::*;
use crate::core::schema::TemporalGraph;

// === Episode CRUD ===

#[tokio::test]
async fn test_insert_episode() {
    let g = TemporalGraph::open_in_memory().await.unwrap();
    let id = g
        .insert_episode(InsertEpisodeParams {
            content: "Alice joined the meeting.",
            timestamp: Utc::now(),
            source_type: Some("transcript"),
            metadata: Some(serde_json::json!({"speaker": "alice"})),
        })
        .await
        .unwrap();
    assert!(id > 0);
}

