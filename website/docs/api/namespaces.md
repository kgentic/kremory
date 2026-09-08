# Namespaces + multi-tenancy


`Namespace { namespace: String, thread: Option<String> }` is the multi-tenant primitive.
Entities are **fully isolated per namespace** — no cross-namespace data leakage.

```rust
use kremory::Namespace;

// Simple namespace
let ns = Namespace::new("tenant-acme");

// Namespace + conversation thread
let ns_thread = Namespace::new("tenant-acme").with_thread("support-ticket-1042");
```

### Default namespace (set once at construction)

`default_namespace` is a `MemoryBuilder` method — reach for `Memory::open(...)` (Tier 2) rather
than `Memory::auto(...)` (Tier 1, a plain `async fn` with no builder chain) when you want a
default namespace set once at construction:

```rust
let mem = Memory::open("./agent.db")
    .with_llm(llm)
    .with_embedder(emb)
    .default_namespace(Namespace::new("user-jim"))
    .await?;

// All subsequent calls use "user-jim" unless overridden
mem.remember("User prefers dark mode").await?;
let ctx = mem.recall("UI preferences").await?;
```

### Per-call namespace override

```rust
// Override default for one specific call
mem.remember("Acme Corp ticket data")
    .in_namespace(Namespace::new("tenant-acme"))
    .await?;
```

### Required namespace (no default set)

If no `default_namespace` is set on `Memory` **and** `.in_namespace()` is not called,
the terminal `.await?` returns `Err(Error::MissingNamespace { request: "remember" })`.
This is a compile-time-visible design choice — the error type is named and matchable:

```rust
match mem.remember("data").await {
    Ok(commit) => { /* ... */ }
    Err(kremory::MemoryError::MissingNamespace { request }) => {
        eprintln!("must call .in_namespace() or set a default_namespace for {request}");
    }
    Err(e) => return Err(e.into()),
}
```

### Multi-tenant SaaS pattern

```rust
// One Memory per process, many namespaces:
let mem = Memory::open("./shared.db")
    .with_llm(llm)
    .with_embedder(emb)
    .await?;

for tenant in ["acme", "globex", "initech"] {
    mem.remember(format!("Tenant {} onboarded", tenant))
        .in_namespace(Namespace::new(tenant))
        .await?;
}

// Each tenant's recall is namespace-scoped — no cross-tenant leakage
let ctx = mem.recall("onboarding status")
    .in_namespace(Namespace::new("acme"))
    .await?;
```

### Namespace policies (v0.1.4+, ADR-029a)

Per-namespace policy controls let callers DECLARE compliance intent (audit-grade,
non-forgettable, non-dream-eligible). v0.1.4 **PERSISTS** the declaration and
emits operational warnings, but does **NOT ENFORCE** the policy on
`dream()` / `forget()` / mutation operations — enforcement lands in v0.1.5+
per ADR-029b. Use the v0.1.4 window to capture intent + validate consumer
ergonomics ahead of enforcement.

```rust
use kremory::{ImmutabilityLevel, Memory, Namespace, NamespacePolicy};

let mem = Memory::open("./shared.db")
    .with_llm(llm)
    .with_embedder(emb)
    .await?;

// Canonical audit-grade preset: AppendOnly + non-forgettable + non-dream-eligible.
let audit = Namespace::new("compliance-log")
    .with_policy(NamespacePolicy::APPEND_ONLY)?;
mem.register_namespace(audit).await?;

// Or build a custom policy via fluent setters.
let custom = NamespacePolicy::new()
    .with_immutability(ImmutabilityLevel::Mutable)
    .with_forgettable(false)
    .with_dream_eligible(true);
let ns = Namespace::new("never-forget").with_policy(custom)?;
mem.register_namespace(ns).await?;
```

`.with_policy(...)`'s `?` is `InvalidPolicyError` — currently one variant,
`IncoherentAppendOnly { policy }`, returned when `immutability: AppendOnly` is combined with
`forgettable=true` or `dream_eligible=true` (AppendOnly requires both `false`, since forget and
dream are both mutations).

**Idempotency + immutability**: calling `register_namespace` with the SAME
policy is `Ok(())` (safe for startup-code re-execution). Calling with a
DIFFERENT policy on an existing namespace surfaces
`Err(MemoryError::Core(Error::NamespacePolicyImmutable { stored, attempted }))`.

**Lazy population**: namespaces observed via the first `remember()` / `recall()`
/ `forget()` / `dream()` call get a default-policy row written automatically.
Call `register_namespace` explicitly at startup for namespaces that need a
non-default policy declared up front.

**Retroactive upgrade**: `Memory::upgrade_namespace_policy(namespace)` monotonically ratchets an
existing namespace's immutability from `Mutable` to `AppendOnly` (ADR-029b Decision 5) — a
**one-way** move; attempting the reverse (`AppendOnly → Mutable`) returns
`Err(MemoryError::Core(Error::NamespacePolicyImmutable { .. }))`. Calling it on an
already-`AppendOnly` namespace is idempotent (`Ok(())`).

```rust
// Ratchet an existing namespace to AppendOnly. Idempotent if already AppendOnly;
// errors on a downgrade attempt or a policy mismatch.
mem.upgrade_namespace_policy(Namespace::new("compliance-log")).await?;
```

**Operational visibility**: every non-default policy registration emits
`tracing::warn!` on target `kremory.namespace` with the marker
`POLICY DECLARED BUT NOT ENFORCED`. Default-policy registrations are silent.

### Custom entity-type registry (per-namespace vocabulary)

Every namespace has an `entity_types` registry (id=0 "Entity" is always the catch-all). The
general-purpose defaults cover common cases; a domain consumer that needs its own vocabulary
(e.g. "Court", "Statute" for a legal use case) seeds it explicitly via `NamespaceSeed`:

```rust
use kremory::{EntityTypeSpec, NamespaceSeed};

// At startup, BEFORE the first `remember()` for this namespace:
let outcome = mem.register_namespace_with_seed(
    Namespace::new("legal-docs"),
    NamespaceSeed::Augment(vec![EntityTypeSpec {
        id: 10,   // consumer ids should be >= 10; 0-9 are the general defaults
        name: "Court".into(),
        description: "A court, tribunal, or judicial body.".into(),
    }]),
).await?;
```

`SeedOutcome::Seeded { rows_written }` on a fresh namespace; `SeedOutcome::AlreadySeeded` on a
namespace that already has rows matching the seed (idempotent, safe for repeated startup calls).
`NamespaceSeed::Replace` is greenfield-only — it fails loudly (`NamespaceRegistrationError::
AlreadyPopulated`) rather than mutate a populated namespace's existing `entity_type_id`s.

To add a type to an ALREADY-populated namespace, use the incremental path,
`Memory::assert_entity_type`, which also pins an entity as `ConsumerPinned` so dream-phase
reclassification never overwrites it:

```rust
use kremory::GraphAssertEntityTypeParams;

mem.assert_entity_type(GraphAssertEntityTypeParams {
    entity_id: "court-of-appeal",
    entity_type_id: 10,
    group_id: None,   // None = default namespace
}).await?;
```

`MemoryBuilder::with_seed_registry(seed)` applies the same seeding at `Memory::open(...)`
construction time, for the default namespace.

---
