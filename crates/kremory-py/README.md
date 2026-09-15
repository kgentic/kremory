# kremory-py

**There is no Python binding for kremory yet.** This directory is a skeleton, not an early
version of one.

## What actually exists here

`src/lib.rs` is one function (`kremory_version()`) that proves this crate's build setup — its
own standalone Cargo workspace (deliberately excluded from the root `kremory` workspace so it
doesn't inherit `panic = "abort"`, which would crash the whole embedded Python interpreter on
any internal Rust panic) — actually compiles and links against the real `kremory` crate. Nothing
else. No PyO3 bindings, no `Memory` class, no methods.

The real binding surface (an estimated 54-108 methods, several weeks of work) has not been
started. If you're looking for a working Python SDK: it doesn't exist. See the
[Rust crate](https://crates.io/crates/kremory) or the [Node.js binding](../kremory-napi/) instead.

## For contributors picking this up

The proven PyO3 bridge pattern to build from lives in a spike, not here — check
`spike/python-binding-2026-09-14/bridge.rs` (repo root) before writing new binding code from
scratch.
