//! Arena allocator primitives for kremory::core.
//!
//! ## Per-episode arena allocation
//!
//! Intermediate strings and
//! structs allocated during a single `add_episode` call are bump-allocated from
//! a `Bump` arena. The arena is reset between episodes — avoiding per-episode
//! heap allocation overhead while keeping the hot path alloc-free.
//!
//! `bumpalo::Bump` survives `.await` on tokio multi-thread runtime — verified
//! empirically, high confidence.
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
