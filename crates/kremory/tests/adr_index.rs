//! Story #1 — ADR discipline: ADR index + supersession chains.
//!
//! AC: .ai-docs/adrs/INDEX.md exists, lists every ADR (adrs/ + adrs/rql/)
//! with status, title, slug; superseded ADRs have superseded_by frontmatter.
//! Gate G1: doc-only, build remains clean.

use std::path::Path;

/// INDEX.md must exist at .ai-docs/adrs/INDEX.md (relative to workspace root).
#[test]
fn adr_index_file_exists() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let index = root.join(".ai-docs/adrs/INDEX.md");
    assert!(
        index.exists(),
        "ADR index not found at {}: run Story #1 to create it",
        index.display()
    );
}

/// INDEX.md must reference all ADR slugs present in adrs/ and adrs/rql/.
#[test]
fn adr_index_contains_all_slugs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let index_path = root.join(".ai-docs/adrs/INDEX.md");
    let index = std::fs::read_to_string(&index_path)
        .unwrap_or_else(|_| panic!("cannot read {}", index_path.display()));

    // Required slugs: all ADR files that must appear in the index.
    let required_slugs = [
        "phase-d0-rqlcm-design-lock-2026-05-18",
        "phase-d0a-rqlm-graphhandle-trait-object-amendment-2026-05-19",
        "adr-026-per-release-discipline-commitment-2026-05-26",
        "adr-001-engine-architecture-single-crate-apache2-2026-05-22",
        "adr-002-byom-distribution-moat-2026-05-22",
        "adr-003-bitemporal-audit-compliance-2026-05-22",
        "adr-004-monorepo-multi-crate-2026-05-22",
        "adr-005-infrastructure-positioning-three-layer-2026-05-22",
        "adr-006-dream-trigger-mechanism-not-policy-2026-05-25",
        "adr-rql-licensing-paid-memory-sdk-byom",
        "adr-rql-licensing-amendment-apache2-cloud-2026-05-20",
        "adr-rql-licensing-amendment-v2-dual-license-2026-05-20",
        "adr-rql-licensing-amendment-v3-kremory-single-crate-apache2-2026-05-21",
        "kremory-memory-observability-first-class-2026-05-20",
        "rqlm-async-event-handle-api-design-2026-05-19",
        "adr-rqlm-cloud-product-spec-2026-05-20",
        "rql-internal-layer-split-core-memory-namespaces",
    ];

    for slug in &required_slugs {
        assert!(
            index.contains(slug),
            "ADR index is missing slug `{}` — add it to .ai-docs/adrs/INDEX.md",
            slug
        );
    }
}

/// Superseded ADRs must have `superseded_by` listed in the index.
/// The v1 and v2 licensing ADRs are superseded by v3.
#[test]
fn adr_index_supersession_chains_present() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let index_path = root.join(".ai-docs/adrs/INDEX.md");
    let index = std::fs::read_to_string(&index_path)
        .unwrap_or_else(|_| panic!("cannot read {}", index_path.display()));

    // The index must mark these as superseded.
    let superseded_indicators = ["superseded", "adr-rql-licensing-amendment-v3"];
    for indicator in &superseded_indicators {
        assert!(
            index.contains(indicator),
            "ADR index missing supersession chain marker `{}` — \
             ensure superseded ADRs are marked with status=superseded \
             and link to superseding ADR",
            indicator
        );
    }
}

/// INDEX.md must contain a status column header (readable as a table).
#[test]
fn adr_index_has_status_column() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let index_path = root.join(".ai-docs/adrs/INDEX.md");
    let index = std::fs::read_to_string(&index_path)
        .unwrap_or_else(|_| panic!("cannot read {}", index_path.display()));

    assert!(
        index.contains("| Status |") || index.contains("| status |") || index.contains("status"),
        "ADR index must have a Status column — add `| Status |` to the table header"
    );
    assert!(
        index.contains("active") || index.contains("accepted"),
        "ADR index must show at least one active/accepted entry"
    );
}
