#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Story #6 — release-please CI workflow.
//!
//! AC: .github/workflows/release-please.yml exists, uses
//! googleapis/release-please-action, triggers on push to main,
//! release-type: rust.
//! Gate G1: file presence check (build doesn't break, doc+ci only).

use std::path::Path;

fn workflow_path() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR = crates/kremory → walk up two levels to workspace root.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    root.join(".github/workflows/release-please.yml")
}

/// The workflow file must exist.
#[test]
fn release_please_workflow_file_exists() {
    let path = workflow_path();
    assert!(
        path.exists(),
        "release-please workflow not found at {}: run Story #6 to create it",
        path.display()
    );
}

/// The workflow must trigger on push to main.
#[test]
fn release_please_triggers_on_push_to_main() {
    let path = workflow_path();
    let content =
        std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("cannot read {}", path.display()));
    assert!(
        content.contains("push") && content.contains("main"),
        "workflow must trigger on push to main; found: {content}"
    );
}

/// The workflow must use googleapis/release-please-action.
#[test]
fn release_please_uses_canonical_action() {
    let path = workflow_path();
    let content =
        std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("cannot read {}", path.display()));
    assert!(
        content.contains("googleapis/release-please-action"),
        "workflow must use googleapis/release-please-action; found: {content}"
    );
}

/// The workflow must declare release-type: rust.
#[test]
fn release_please_release_type_rust() {
    let path = workflow_path();
    let content =
        std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("cannot read {}", path.display()));
    assert!(
        content.contains("release-type") && content.contains("rust"),
        "workflow must set release-type: rust; found: {content}"
    );
}
