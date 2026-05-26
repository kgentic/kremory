//! G12 acceptance tests — named struct variants on `RqlError`.
//!
//! Each test constructs a new struct variant, verifies field extraction via
//! pattern matching, asserts the `Display` impl renders readably, and confirms
//! the type implements `std::error::Error`.
//!
//! Story #155 — must pass before #156 (downstream Result<_, Error> signatures
//! depend on final taxonomy).

use kremory::core::error::RqlError;

// ── helpers ──────────────────────────────────────────────────────────────────

fn is_std_error<E: std::error::Error>(_: &E) {}

// ── 1. InsertReturnedNoRowId ──────────────────────────────────────────────────

#[test]
fn insert_returned_no_row_id_fields_extractable() {
    let e = RqlError::InsertReturnedNoRowId {
        operation: "insert_fact",
    };
    match &e {
        RqlError::InsertReturnedNoRowId { operation } => {
            assert_eq!(*operation, "insert_fact");
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn insert_returned_no_row_id_display_contains_operation() {
    let e = RqlError::InsertReturnedNoRowId {
        operation: "insert_episode",
    };
    let msg = e.to_string();
    assert!(
        msg.contains("insert_episode"),
        "Display must mention the operation; got: {msg}"
    );
}

#[test]
fn insert_returned_no_row_id_is_std_error() {
    let e = RqlError::InsertReturnedNoRowId {
        operation: "insert_fact",
    };
    is_std_error(&e);
}

// ── 2. EmbeddingDimZero ───────────────────────────────────────────────────────

#[test]
fn embedding_dim_zero_constructs() {
    let e = RqlError::EmbeddingDimZero;
    match e {
        RqlError::EmbeddingDimZero => {}
        _ => panic!("wrong variant"),
    }
}

#[test]
fn embedding_dim_zero_display_is_readable() {
    let e = RqlError::EmbeddingDimZero;
    let msg = e.to_string();
    // Must mention "embedding" or "dim" or "zero" — something diagnostic
    let lower = msg.to_lowercase();
    assert!(
        lower.contains("embedding") || lower.contains("dim"),
        "Display must mention embedding/dim; got: {msg}"
    );
}

#[test]
fn embedding_dim_zero_is_std_error() {
    let e = RqlError::EmbeddingDimZero;
    is_std_error(&e);
}

// ── 3. ExtractionStage ───────────────────────────────────────────────────────

#[test]
fn extraction_stage_fields_extractable() {
    let e = RqlError::ExtractionStage {
        stage: "entities".to_string(),
        detail: "LLM timed out".to_string(),
    };
    match &e {
        RqlError::ExtractionStage { stage, detail } => {
            assert_eq!(stage, "entities");
            assert_eq!(detail, "LLM timed out");
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn extraction_stage_display_contains_stage_and_detail() {
    let e = RqlError::ExtractionStage {
        stage: "triplets".to_string(),
        detail: "parse error".to_string(),
    };
    let msg = e.to_string();
    assert!(
        msg.contains("triplets"),
        "Display must contain stage; got: {msg}"
    );
    assert!(
        msg.contains("parse error"),
        "Display must contain detail; got: {msg}"
    );
}

#[test]
fn extraction_stage_is_std_error() {
    let e = RqlError::ExtractionStage {
        stage: "relations".to_string(),
        detail: "upstream err".to_string(),
    };
    is_std_error(&e);
}

// ── 4. WeightSumInvalid ──────────────────────────────────────────────────────

#[test]
fn weight_sum_invalid_fields_extractable() {
    let e = RqlError::WeightSumInvalid {
        bm25: 0.8,
        vector: 0.5,
    };
    match &e {
        RqlError::WeightSumInvalid { bm25, vector } => {
            assert!((*bm25 - 0.8).abs() < f64::EPSILON);
            assert!((*vector - 0.5).abs() < f64::EPSILON);
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn weight_sum_invalid_display_contains_weights() {
    let e = RqlError::WeightSumInvalid {
        bm25: 0.6,
        vector: 0.6,
    };
    let msg = e.to_string();
    // Must mention both weights somehow (bm25 / vector)
    assert!(
        msg.contains("0.6") || msg.contains("bm25") || msg.contains("vector"),
        "Display must reference the weight values; got: {msg}"
    );
}

#[test]
fn weight_sum_invalid_is_std_error() {
    let e = RqlError::WeightSumInvalid {
        bm25: 0.3,
        vector: 0.4,
    };
    is_std_error(&e);
}

// ── 5. TokenWindowInvalid ─────────────────────────────────────────────────────

#[test]
fn token_window_invalid_fields_extractable() {
    let e = RqlError::TokenWindowInvalid { got: 100, min: 300 };
    match &e {
        RqlError::TokenWindowInvalid { got, min } => {
            assert_eq!(*got, 100);
            assert_eq!(*min, 300);
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn token_window_invalid_display_contains_values() {
    let e = RqlError::TokenWindowInvalid { got: 50, min: 200 };
    let msg = e.to_string();
    assert!(
        msg.contains("50") && msg.contains("200"),
        "Display must contain both got and min; got: {msg}"
    );
}

#[test]
fn token_window_invalid_is_std_error() {
    let e = RqlError::TokenWindowInvalid { got: 10, min: 100 };
    is_std_error(&e);
}
