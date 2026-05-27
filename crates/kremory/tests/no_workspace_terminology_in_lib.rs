//! A.7 gating test — ensures `WorkspaceScope` is fully removed from the substrate src.
//!
//! The substrate must be consumer-neutral. `WorkspaceScope` was the-host-application-flavored
//! vocabulary leaked into the public API. Industry standard for agent memory
//! partitioning is `Namespace` (Mem0, Zep prior art). This test fails the build
//! the moment any `WorkspaceScope` reference creeps back into `crates/kremory/src/`.

#[test]
fn no_workspace_scope_in_substrate_src() {
    use std::process::Command;
    let output = Command::new("grep")
        .args(["-rn", "WorkspaceScope", "crates/kremory/src/"])
        .output()
        .expect("grep failed");
    assert!(
        output.stdout.is_empty(),
        "WorkspaceScope still present in substrate src — rename to Namespace:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}
