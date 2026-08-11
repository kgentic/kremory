//! TD-201 cause-fix mechanical proof — NOT committed to the crate, scratch only.
//!
//! Demonstrates the load-bearing claim: after replacing the two hand-synced
//! declaration sites (`RESERVED_PREDICATE_*` const + `RESERVED_PREDICATES`
//! array) with a single `reserved_predicates!` macro invocation, adding a
//! second reserved predicate is a ONE-LINE, ZERO-OTHER-EDITS change, and it
//! is impossible to declare a predicate via the macro that is absent from
//! `RESERVED_PREDICATES`.
//!
//! This is an exact copy of the macro + fn shape now living in
//! `crates/kremory/src/core/disambiguation/mod.rs`, with a SECOND predicate
//! (`RESERVED_PREDICATE_SPIKE_PROOF`) added to the invocation to prove the
//! single-site property, per [[mechanical-compile-spike-beats-paper-review]].
//! Run with: `rustc --edition 2021 spike/td201_reserved_predicates_macro_binding.rs -o /tmp/td201_spike && /tmp/td201_spike`
//!
//! Revert instructions for the real module: none needed — this file is not
//! part of the crate and touches nothing in `crates/`.

macro_rules! reserved_predicates {
    ($($(#[$meta:meta])* $konst:ident => $lit:literal),+ $(,)?) => {
        $(
            $(#[$meta])*
            pub const $konst: &str = $lit;
        )+

        pub const RESERVED_PREDICATES: &[&str] = &[$($konst),+];
    };
}

reserved_predicates! {
    /// Original predicate (mirrors the shipped module).
    RESERVED_PREDICATE_POTENTIAL_ALIAS => "potential_alias",
    /// PROOF-ONLY second predicate — added with ZERO other edits to this
    /// file besides this one macro-invocation line, to demonstrate the
    /// single-declaration-site property mechanically rather than by
    /// argument.
    RESERVED_PREDICATE_SPIKE_PROOF => "spike_proof_second_predicate",
}

pub fn is_reserved_predicate(predicate: &str) -> bool {
    RESERVED_PREDICATES.contains(&predicate)
}

fn main() {
    // 1. The single macro invocation above generated BOTH consts AND folded
    //    BOTH into RESERVED_PREDICATES, with no second list anywhere in this
    //    file for them to drift against.
    assert_eq!(
        RESERVED_PREDICATES.len(),
        2,
        "expected exactly 2 reserved predicates after adding the second \
         macro-invocation line with zero other edits"
    );
    assert_eq!(RESERVED_PREDICATE_POTENTIAL_ALIAS, "potential_alias");
    assert_eq!(RESERVED_PREDICATE_SPIKE_PROOF, "spike_proof_second_predicate");

    // 2. Both new consts are reachable through RESERVED_PREDICATES...
    assert!(RESERVED_PREDICATES.contains(&RESERVED_PREDICATE_POTENTIAL_ALIAS));
    assert!(RESERVED_PREDICATES.contains(&RESERVED_PREDICATE_SPIKE_PROOF));

    // 3. ...and therefore through is_reserved_predicate(), the actual
    //    consumer-facing filter. This is the property that matters: adding
    //    a predicate via the macro makes it filterable with ZERO additional
    //    wiring anywhere else.
    assert!(is_reserved_predicate("potential_alias"));
    assert!(is_reserved_predicate("spike_proof_second_predicate"));
    assert!(!is_reserved_predicate("not_a_reserved_predicate"));

    println!(
        "PASS: single macro invocation, {} predicates, zero secondary lists, \
         zero drift possible.",
        RESERVED_PREDICATES.len()
    );

    // 4. Named, not proven-away: the residual limit TD-201 already
    //    documents is real and this macro does NOT close it. The following
    //    would ALSO compile — a raw `pub const RESERVED_PREDICATE_BYPASS`
    //    declared OUTSIDE the macro invocation, silently absent from
    //    RESERVED_PREDICATES and therefore never filtered by
    //    is_reserved_predicate(). Left commented out (not executed) because
    //    demonstrating a bypass compiling is not the same as needing to run
    //    it — the point is only that nothing in the type system prevents
    //    writing it:
    //
    //   pub const RESERVED_PREDICATE_BYPASS: &str = "bypass_the_macro";
    //   // ^ compiles fine; RESERVED_PREDICATES.len() stays 2, not 3 — this
    //   //   const is real Rust but invisible to is_reserved_predicate().
    //
    // This is the SAME residual documented in the module's own doc comment
    // and in TD-201: the `RESERVED_PREDICATE_*` naming convention plus "use
    // the macro" is a discipline, not a compiler-enforced law. What the
    // macro DOES remove is the failure mode TD-197 actually hit: declaring
    // a predicate the *intended* way (via the macro) and forgetting a
    // second manual step. That failure mode is now unrepresentable.
}
