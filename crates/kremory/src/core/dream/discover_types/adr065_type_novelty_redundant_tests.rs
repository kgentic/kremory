// ─── type_novelty_is_redundant unit tests ────────────────────────────────────

use super::*;

fn verdict(is_same: bool, confidence: f32) -> Option<IdentityVerdictItem> {
    Some(IdentityVerdictItem {
        pair_id: 0,
        is_same_entity: is_same,
        confidence,
        reasoning: String::new(),
    })
}

/// is_same + confidence at/above the floor → REDUNDANT (the fix's core case:
/// a correct confident `true` verdict is no longer downgraded by write_gate
/// Row 6). This is exactly the s2-001 Firm/Company shape.
#[test]
fn is_same_high_conf_is_redundant() {
    assert!(type_novelty_is_redundant(&verdict(
        true,
        LLM_VERIFY_CONFIDENCE_FLOOR
    )));
    assert!(type_novelty_is_redundant(&verdict(true, 1.0)));
}

/// is_same but confidence BELOW the floor → NOT redundant (conservative
/// accept; weak agreement must not reject a new type).
#[test]
fn is_same_below_floor_not_redundant() {
    assert!(!type_novelty_is_redundant(&verdict(
        true,
        LLM_VERIFY_CONFIDENCE_FLOOR - 0.01
    )));
}

/// LLM says distinct → NOT redundant (EDC false-reject-prevention: accept).
#[test]
fn not_same_not_redundant() {
    assert!(!type_novelty_is_redundant(&verdict(false, 1.0)));
}

/// No verdict (LLM/parse failure) → NOT redundant (conservative accept —
/// a failed adjudication must not silently reject a proposal).
#[test]
fn no_verdict_not_redundant() {
    assert!(!type_novelty_is_redundant(&None));
}
