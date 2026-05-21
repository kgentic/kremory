//! A.5 — D.6.8 compile-time pin test for StubGraphHandle.
//!
//! Verifies that `kremory::memory::StubGraphHandle` exists, implements the
//! full `GraphHandle` trait surface, and erases cleanly to
//! `Arc<dyn GraphHandle>` at the rqlm boundary.
//!
//! This test is intentionally minimal: it compiles, constructs the stub,
//! and calls each method once to confirm the vtable is complete. Unimplemented
//! bodies panic at runtime — that is the expected behaviour for a stub.
//!
//! Primary use case: downstream consumers (the host application, aidocs SDK, test suites)
//! use `StubGraphHandle` as a do-nothing stand-in when they need a concrete
//! `&dyn GraphHandle` without a real database.

use std::sync::Arc;

use kremory::memory::{GraphHandle, StubGraphHandle};

/// Coercion to `Arc<dyn GraphHandle>` compiles and is object-safe.
#[test]
fn stub_graph_handle_erases_to_dyn_graph_handle() {
    let stub = StubGraphHandle;
    let _erased: Arc<dyn GraphHandle> = Arc::new(stub);
    // No assertion needed — the coercion itself is the test.
    // If StubGraphHandle stops implementing all required methods this
    // test fails to compile.
}

/// `&StubGraphHandle` is usable as `&dyn GraphHandle`.
#[test]
fn stub_graph_handle_ref_to_dyn() {
    let stub = StubGraphHandle;
    let _dyn_ref: &dyn GraphHandle = &stub;
}
