# Running the test suite

## Fast local loop (recommended)

kremory uses [cargo-nextest](https://nexte.st) as the local test runner. It runs
all test binaries in one global parallel pool, which on this suite is ~16x faster
than `cargo test` (measured 568s → 35s).

```bash
cargo nextest run -p kremory      # unit + integration tests (~35s)
cargo test --doc -p kremory       # doctests — nextest does NOT run these
```

**Run both.** nextest covers unit + integration tests; doctests are only executed
by `cargo test --doc`. Running `cargo nextest run` alone **silently skips doctests**.

Install nextest once: `cargo install cargo-nextest --locked` (or the prebuilt
binary from [https://get.nexte.st](https://get.nexte.st)).

## Full workspace

```bash
cargo nextest run --workspace && cargo test --workspace --doc
```

The cross-crate `kremory-napi` parity gate is only visible workspace-wide —
`cargo nextest run -p kremory` alone will not run it.

## Config

`.config/nextest.toml` holds the default profile. It documents a commented
`test-groups` scaffold for serializing resource-contended tests (ollama / ner /
libSQL single-writer) to enable if feature-profile flakes appear under nextest's
higher parallelism.

## Why not just `cargo test`?

`cargo test` runs the 133 test binaries **serially** (parallelizing only within
each binary), which is where the ~9.5-minute wall-clock came from — it is
test *execution*, not linking (incremental relink of all 133 binaries is ~12s).
nextest's global parallel pool is the lever. Full analysis in the spec linked above.
