//! Arena allocator primitives for kremory::core.
//!
//! ## P4 — per-episode arena allocation
//!
//! Per architecture spec §2.4.B (P4 confidence gate): intermediate strings and
//! structs allocated during a single `add_episode` call are bump-allocated from
//! a `Bump` arena. The arena is reset between episodes — avoiding per-episode
//! heap allocation overhead while keeping the hot path alloc-free.
//!
//! A.0 outcome: `bumpalo::Bump` survives `.await` on tokio multi-thread runtime
//! (K7 = HIGH confidence). B.1 wires `Bump` through the arena module.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use kremory::core::arena::Bump;
//!
//! let mut bump = Bump::new();
//! let s: &str = bump.alloc_str("per-episode string");
//! // ... process episode ...
//! bump.reset(); // reclaim memory for next episode
//! ```

/// Re-export of `bumpalo::Bump` — the arena allocator used for per-episode
/// intermediate allocation. Re-exported so consumers import from `kremory::core::arena`
/// rather than depending on `bumpalo` directly.
pub use bumpalo::Bump;
