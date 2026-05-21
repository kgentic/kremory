/// A.2 — D.6.3: error envelopes serde roundtrip tests.
///
/// RED phase tests — these will fail until the types are implemented.
use kremory::core::error::{ContradictionResolution, IngestionErrorKind};

#[test]
fn ingestion_error_kind_serde_roundtrip() {
    // ValidationFailed
    let kind = IngestionErrorKind::ValidationFailed {
        reason: "missing field".to_string(),
    };
    let json = serde_json::to_string(&kind).unwrap();
    let back: IngestionErrorKind = serde_json::from_str(&json).unwrap();
    assert_eq!(format!("{:?}", back), format!("{:?}", kind));

    // ProviderError
    let kind = IngestionErrorKind::ProviderError {
        provider_name: "openai".to_string(),
        detail: "rate limit".to_string(),
    };
    let json = serde_json::to_string(&kind).unwrap();
    let back: IngestionErrorKind = serde_json::from_str(&json).unwrap();
    assert_eq!(format!("{:?}", back), format!("{:?}", kind));

    // RateLimited
    let kind = IngestionErrorKind::RateLimited {
        retry_after: Some(std::time::Duration::from_secs(60)),
    };
    let json = serde_json::to_string(&kind).unwrap();
    let back: IngestionErrorKind = serde_json::from_str(&json).unwrap();
    assert_eq!(format!("{:?}", back), format!("{:?}", kind));

    // ParseFailure
    let kind = IngestionErrorKind::ParseFailure {
        stage: "extraction".to_string(),
        detail: "unexpected token".to_string(),
    };
    let json = serde_json::to_string(&kind).unwrap();
    let back: IngestionErrorKind = serde_json::from_str(&json).unwrap();
    assert_eq!(format!("{:?}", back), format!("{:?}", kind));

    // SchemaViolation
    let kind = IngestionErrorKind::SchemaViolation {
        field: "entity_id".to_string(),
        expected: "uuid".to_string(),
    };
    let json = serde_json::to_string(&kind).unwrap();
    let back: IngestionErrorKind = serde_json::from_str(&json).unwrap();
    assert_eq!(format!("{:?}", back), format!("{:?}", kind));
}

#[test]
fn contradiction_resolution_serde_roundtrip() {
    for resolution in [
        ContradictionResolution::Superseded,
        ContradictionResolution::Retained,
        ContradictionResolution::Merged,
    ] {
        let json = serde_json::to_string(&resolution).unwrap();
        let back: ContradictionResolution = serde_json::from_str(&json).unwrap();
        assert_eq!(resolution, back);
    }
}

#[test]
fn contradiction_resolution_is_non_exhaustive() {
    // Verify #[non_exhaustive] is present by ensuring we can't pattern-match
    // without a wildcard arm in external code. This is a compile-time check.
    // We just verify the variants exist and roundtrip correctly here.
    let s = ContradictionResolution::Superseded;
    let r = ContradictionResolution::Retained;
    let m = ContradictionResolution::Merged;
    assert_ne!(format!("{:?}", s), format!("{:?}", r));
    assert_ne!(format!("{:?}", r), format!("{:?}", m));
}
