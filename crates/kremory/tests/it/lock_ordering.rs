#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Story #229 — Lock ordering documented.
//!
//! AC: module-level doc comment in `core/mod.rs` documents lock acquisition
//!     ordering to prevent deadlocks as new concurrent state is added.
//! Gate G1: cargo build -p kremory --all-features → 0 errors.

/// Verify that the lock-ordering invariant is documented in `core/mod.rs`.
///
/// Checks for the canonical marker string that signals the ordering comment
/// exists. Any future reader (human or tooling) can grep for this string to
/// locate the authoritative lock-ordering specification.
#[test]
fn lock_ordering_comment_present_in_core_mod() {
    use std::path::Path;

    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let core_mod = root.join("crates/kremory/src/core/mod.rs");
    let content = std::fs::read_to_string(&core_mod)
        .unwrap_or_else(|_| panic!("cannot read {}", core_mod.display()));

    assert!(
        content.contains("# Lock ordering"),
        "core/mod.rs must contain a '# Lock ordering' doc comment section \
         documenting mutex acquisition order (Story #229)"
    );
}

/// Verify that the lock ordering comment names the ADR-022 write serialiser.
#[test]
fn lock_ordering_references_write_serialiser() {
    use std::path::Path;

    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let core_mod = root.join("crates/kremory/src/core/mod.rs");
    let content = std::fs::read_to_string(&core_mod)
        .unwrap_or_else(|_| panic!("cannot read {}", core_mod.display()));

    assert!(
        content.contains("write_lock") || content.contains("ADR-022"),
        "core/mod.rs lock ordering section must reference the ADR-022 write \
         serialiser (write_lock) (Story #229)"
    );
}
