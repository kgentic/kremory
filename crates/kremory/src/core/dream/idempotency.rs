// ---------------------------------------------------------------------------
// Dream-pass idempotency key — canonical entity view + content hash.
//
// COMPILE-SPIKE (v0.2.4 Phase 3, readiness-gate fix 2026-06-16): the readiness
// gate flipped FAIL→READY contingent on `canonical_entity_view` being the first
// compile-spike of the sprint (per CLAUDE.md Rule 23 — compile-spike beats paper
// review). This module is that spike: it defines the canonical view + hash that
// the idempotency-key tuple `(pass_name, entity_id, content_hash)` is built from.
//
// ## Why a content hash (R-10)
//
// A dream pass must skip an entity it has already successfully processed in the
// same state. "Same state" excludes cosmetic / audit-only mutations: a row whose
// `updated_at` ticked or whose `access_count` incremented is NOT a new state and
// MUST hash identically, otherwise the idempotency guard never fires and every
// pass re-processes every entity forever (R-10 stop condition: duplicate
// processing > 0).
//
// ## Canonical field set — REAL `Entity` struct, not the spec's idealised list
//
// The v0.2.4 impl-spec §3 named the field set `{id, name, entity_type_id,
// group_id, source_tier, source_id}`. Verified against `core/schema.rs::Entity`
// (Rule 10) THREE of those do not exist on the in-memory struct:
//   - `name`        → the struct field is `label`
//   - `source_tier` → a DB column (migrations 012/013) but NOT an `Entity` field;
//                     unreachable from `&Entity` without a separate query
//   - `source_id`   → exists on `Episode`, not on `Entity`
//
// The defensible identity field set that (a) exists on `&Entity` and (b) captures
// load-bearing state (NOT audit/mutable churn) is:
//
//   { id, label, entity_type_id, group_id }
//
// Excluded as audit/mutable-churn: `recorded_at`, `updated_at`, `access_count`.
// Excluded as non-identity payload: `properties` (free-form JSON whose mutation
// does not change which entity this is). If a future pass needs `source_tier` in
// the hash, the canonical view must take a richer projection than `&Entity` — a
// design change tracked for the Phase 3 integration, not this spike.
// ---------------------------------------------------------------------------

use sha2::{Digest, Sha256};

use crate::core::schema::Entity;

/// Unit separator (0x1F) between canonical fields. Same delimiter the VCR
/// fingerprint uses (`provider/record_replay.rs`) — keeps field boundaries
/// unambiguous so two distinct field layouts can never collide.
const FIELD_SEP: char = '\u{1F}';

/// Sentinel emitted for a `None` `group_id` so that `Some("")` and `None`
/// produce DIFFERENT canonical views (otherwise both would serialise to an
/// empty value and collide).
const NONE_SENTINEL: &str = "\u{0}none";

/// The exact, ordered field set hashed by [`canonical_entity_view`].
///
/// Single source of truth: the serialisation order in [`canonical_entity_view`]
/// MUST match this slice element-for-element. The `canonical_view_fields_match`
/// test locks the two together so the list cannot silently drift from the impl.
///
/// Audit/mutable fields (`recorded_at`, `updated_at`, `access_count`) and the
/// free-form `properties` payload are deliberately ABSENT — see module docs.
///
/// **Test-only**: this function exists solely to expose the canonical field list
/// so that `canonical_view_fields_match` can assert the serialisation order
/// matches this declaration. It has no production call site — `content_hash`
/// calls `canonical_entity_view` directly. Gated `#[cfg(test)]` to eliminate
/// the dead-code lint without a band-aid `#[allow]`.
#[cfg(test)]
pub fn canonical_entity_view_fields() -> &'static [&'static str] {
    &["id", "label", "entity_type_id", "group_id"]
}

/// Deterministic serialisation of an entity's load-bearing identity state.
///
/// Built by hand field-by-field with a 0x1F separator rather than via
/// `serde_json` object serialisation: serde_json key-ordering is only stable
/// here because the `preserve_order` feature is off (documented fragile
/// assumption in `record_replay.rs`). A hand-built delimited string is
/// deterministic UNCONDITIONALLY, independent of any future transitive feature
/// unification.
///
/// Excludes audit/mutable fields so cosmetic timestamp / access-count changes do
/// NOT change the view (R-10). The field set is enumerated by
/// [`canonical_entity_view_fields`].
pub fn canonical_entity_view(entity: &Entity) -> String {
    let group = entity.group_id.as_deref().unwrap_or(NONE_SENTINEL);
    // Order MUST match `canonical_entity_view_fields()`.
    format!(
        "id={id}{sep}label={label}{sep}entity_type_id={tid}{sep}group_id={group}",
        id = entity.id,
        label = entity.label,
        tid = entity.entity_type_id,
        group = group,
        sep = FIELD_SEP,
    )
}

/// SHA-256 hex digest of the [`canonical_entity_view`]. This is the
/// `content_hash` component of the dream-pass idempotency key
/// `(pass_name, entity_id, content_hash)`.
pub fn content_hash(entity: &Entity) -> String {
    let view = canonical_entity_view(entity);
    let digest = Sha256::digest(view.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write as _;
        // Writing to a String is infallible; matches record_replay.rs hex loop.
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone as _, Utc};

    // Test helper: Rule-5 exempt per clippy.toml (test helpers may carry a
    // documented too_many_arguments allow); TD-042 args-as-object targets `src/`
    // production fns, not `#[cfg(test)]` builders.
    #[allow(clippy::too_many_arguments)]
    fn entity(id: &str, label: &str, type_id: u32, group: Option<&str>) -> Entity {
        Entity {
            id: id.to_string(),
            label: label.to_string(),
            entity_type_id: type_id,
            properties: serde_json::json!({"any": "payload"}),
            recorded_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
            updated_at: None,
            group_id: group.map(str::to_string),
            access_count: 0,
        }
    }

    #[test]
    fn canonical_view_is_deterministic() {
        let e = entity("ent-1", "Person", 3, Some("default"));
        assert_eq!(canonical_entity_view(&e), canonical_entity_view(&e));
        assert_eq!(content_hash(&e), content_hash(&e));
    }

    #[test]
    fn audit_and_mutable_fields_excluded_from_hash() {
        // R-10: rows differing ONLY in audit/mutable fields hash identically.
        let base = entity("ent-1", "Person", 3, Some("default"));
        let mut churned = base.clone();
        churned.recorded_at = Utc.timestamp_opt(1_999_999_999, 0).unwrap();
        churned.updated_at = Some(Utc.timestamp_opt(1_888_888_888, 0).unwrap());
        churned.access_count = 9_999;
        churned.properties = serde_json::json!({"completely": "different", "n": 42});

        assert_eq!(
            content_hash(&base),
            content_hash(&churned),
            "audit/mutable/properties churn must NOT change the idempotency hash (R-10)"
        );
    }

    #[test]
    fn identity_changes_change_the_hash() {
        let base = entity("ent-1", "Person", 3, Some("default"));
        // Each identity field, flipped one at a time, must change the hash.
        let diff_id = entity("ent-2", "Person", 3, Some("default"));
        let diff_label = entity("ent-1", "Place", 3, Some("default"));
        let diff_type = entity("ent-1", "Person", 4, Some("default"));
        let diff_group = entity("ent-1", "Person", 3, Some("other"));
        for other in [&diff_id, &diff_label, &diff_type, &diff_group] {
            assert_ne!(
                content_hash(&base),
                content_hash(other),
                "an identity-field change must change the hash"
            );
        }
    }

    #[test]
    fn none_group_differs_from_empty_group() {
        // `None` and `Some("")` must NOT collide.
        let none_group = entity("ent-1", "Person", 3, None);
        let empty_group = entity("ent-1", "Person", 3, Some(""));
        assert_ne!(content_hash(&none_group), content_hash(&empty_group));
    }

    #[test]
    fn canonical_view_fields_match_serialisation() {
        // Lock the SoT field list to the actual serialised order/content.
        let e = entity("X", "Y", 7, Some("G"));
        let view = canonical_entity_view(&e);
        let fields = canonical_entity_view_fields();
        assert_eq!(fields, &["id", "label", "entity_type_id", "group_id"]);
        // Every declared field must appear as a `field=` segment, in order.
        let segments: Vec<&str> = view.split(FIELD_SEP).collect();
        assert_eq!(segments.len(), fields.len());
        for (seg, field) in segments.iter().zip(fields.iter()) {
            assert!(
                seg.starts_with(&format!("{field}=")),
                "segment {seg:?} must declare field {field:?}"
            );
        }
    }

    #[test]
    fn hash_is_sha256_hex_length() {
        let e = entity("ent-1", "Person", 3, Some("default"));
        let h = content_hash(&e);
        assert_eq!(h.len(), 64, "sha256 hex is 64 chars");
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
