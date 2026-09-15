//! Skeleton only — proves the crate's location/workspace-exclusion/profile
//! setup (Next Action #3) compiles and links against `kremory` for real,
//! WITHOUT starting the actual PyO3 binding surface (Next Action #4: ~54
//! awaitable entry points, a separate multi-week effort with its own
//! generating-macro design problem — see
//! `spike/python-binding-2026-09-14/README.md` §"Size").
//!
//! The proven PyO3 bridge to copy when that work starts is
//! `spike/python-binding-2026-09-14/bridge.rs` — do not re-derive it.

/// Proves the path dependency on `kremory` actually resolves and a real
/// substrate type is reachable from this crate — not just that an empty
/// crate compiles.
pub fn kremory_version() -> &'static str {
    // `kremory::Memory` is the facade entry point every real binding builds
    // on; referencing it (rather than a leaf type) is the more honest link
    // proof for a crate whose whole job is to wrap that type.
    let _type_check: fn() -> Option<kremory::Memory> = || None;
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_against_kremory() {
        assert!(!kremory_version().is_empty());
    }
}
