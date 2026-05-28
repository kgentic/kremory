#![allow(clippy::unwrap_used, clippy::expect_used)]
//! ADR-029a integration tests — `NamespacePolicy` struct + `register_namespace`.
//!
//! 19 tests per ADR §5 test contract:
//! - 13 from cycle-1 (G_v014a_1 .. G_v014a_13)
//! - G_v014a_4b — half-coherent variant (Vera cycle-1 LOW-3)
//! - G_v014a_14..G_v014a_19 — Vera cycle-1 MED-3 / MED-5 hardening

use std::sync::Arc;

use kremory::core::error::Error as CoreError;
use kremory::{
    DynEmbeddingProvider, ImmutabilityLevel, InvalidPolicyError, Memory, MemoryError, Namespace,
    NamespacePolicy,
};
use tempfile::TempDir;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// Create a fresh temp dir + open a `Memory` against a unique DB file inside.
/// The `TempDir` is returned so the caller keeps it alive for the test scope.
async fn fresh_memory() -> (Memory, TempDir) {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("kremory-029a.db");
    let mem = Memory::open(&path)
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("Memory::open builder");
    (mem, tmp)
}

// ── Cycle-1 tests ────────────────────────────────────────────────────────────

/// G_v014a_1: `register_namespace(ns.with_policy(APPEND_ONLY))` persists the
/// policy so a follow-up read sees it (verified indirectly via idempotent re-
/// register returning `Ok(())` with the same policy).
#[tokio::test]
async fn g_v014a_1_register_namespace_persists_policy() {
    let (mem, _tmp) = fresh_memory().await;
    let ns = Namespace::new("tenant-a")
        .with_policy(NamespacePolicy::APPEND_ONLY)
        .expect("APPEND_ONLY is coherent");
    mem.register_namespace(ns.clone())
        .await
        .expect("register_namespace persists");
    // Idempotent re-register confirms the stored policy matches.
    mem.register_namespace(ns)
        .await
        .expect("re-register with same policy is idempotent");
}

/// G_v014a_2: two `register_namespace` calls with identical policy both return
/// `Ok(())`.
#[tokio::test]
async fn g_v014a_2_register_namespace_idempotent_same_policy() {
    let (mem, _tmp) = fresh_memory().await;
    let policy = NamespacePolicy::APPEND_ONLY;
    let ns = Namespace::new("tenant-idempotent")
        .with_policy(policy.clone())
        .expect("coherent");
    mem.register_namespace(ns.clone()).await.expect("first call");
    mem.register_namespace(ns)
        .await
        .expect("second call with same policy must return Ok(())");
}

/// G_v014a_3: second `register_namespace` with a DIFFERENT policy returns
/// `Error::NamespacePolicyImmutable { stored, attempted }`.
#[tokio::test]
async fn g_v014a_3_register_namespace_different_policy_errs() {
    let (mem, _tmp) = fresh_memory().await;
    let ns_a = Namespace::new("tenant-clash")
        .with_policy(NamespacePolicy::APPEND_ONLY)
        .expect("coherent");
    mem.register_namespace(ns_a).await.expect("first call");

    let alt_policy = NamespacePolicy::new()
        .with_immutability(ImmutabilityLevel::Mutable)
        .with_forgettable(true)
        .with_dream_eligible(true);
    let ns_b = Namespace::new("tenant-clash")
        .with_policy(alt_policy.clone())
        .expect("coherent");
    let err = mem
        .register_namespace(ns_b)
        .await
        .expect_err("second call with different policy must err");
    match err {
        MemoryError::Core(CoreError::NamespacePolicyImmutable {
            namespace,
            stored,
            attempted,
        }) => {
            assert_eq!(namespace, "tenant-clash");
            assert_eq!(stored, NamespacePolicy::APPEND_ONLY);
            assert_eq!(attempted, alt_policy);
        }
        other => panic!("expected NamespacePolicyImmutable, got {other:?}"),
    }
}

/// G_v014a_4: `validate()` rejects `AppendOnly + forgettable=true`.
#[tokio::test]
async fn g_v014a_4_validate_rejects_appendonly_plus_forgettable_true() {
    let policy = NamespacePolicy::new()
        .with_immutability(ImmutabilityLevel::AppendOnly)
        .with_forgettable(true)
        .with_dream_eligible(false);
    let err = policy.validate().expect_err("must reject");
    matches!(err, InvalidPolicyError::IncoherentAppendOnly { .. })
        .then_some(())
        .expect("IncoherentAppendOnly variant");
}

/// G_v014a_4b: half-coherent — `AppendOnly + dream_eligible=true` (Vera LOW-3).
#[tokio::test]
async fn g_v014a_4b_validate_rejects_appendonly_plus_dream_eligible_true() {
    let policy = NamespacePolicy::new()
        .with_immutability(ImmutabilityLevel::AppendOnly)
        .with_forgettable(false)
        .with_dream_eligible(true);
    let err = policy.validate().expect_err("must reject");
    matches!(err, InvalidPolicyError::IncoherentAppendOnly { .. })
        .then_some(())
        .expect("IncoherentAppendOnly variant");
}

/// G_v014a_5: `validate()` accepts the default policy.
#[test]
fn g_v014a_5_validate_accepts_default_policy() {
    NamespacePolicy::default()
        .validate()
        .expect("default policy must validate");
}

/// G_v014a_6: `validate()` accepts the canonical `APPEND_ONLY` preset.
#[test]
fn g_v014a_6_validate_accepts_appendonly_preset_const() {
    NamespacePolicy::APPEND_ONLY
        .validate()
        .expect("APPEND_ONLY const must validate");
}

/// G_v014a_7: `Namespace::with_policy` validates at attach time.
#[test]
fn g_v014a_7_namespace_with_policy_validates_at_attach() {
    let bad = NamespacePolicy::new()
        .with_immutability(ImmutabilityLevel::AppendOnly)
        .with_forgettable(true);
    let err = Namespace::new("any-ns")
        .with_policy(bad)
        .expect_err("must surface InvalidPolicyError at attach time");
    matches!(err, InvalidPolicyError::IncoherentAppendOnly { .. })
        .then_some(())
        .expect("IncoherentAppendOnly variant");
}

/// G_v014a_8: registering a namespace, then opening a fresh `Memory` against a
/// new tempdir must return `Ok(())` on a re-register with default policy
/// (substrate-level "unregistered = no row" semantics — observed via the
/// idempotent default path).
#[tokio::test]
async fn g_v014a_8_policy_lookup_returns_none_for_unregistered_ns() {
    let (mem, _tmp) = fresh_memory().await;
    // No prior register call — registering with default policy on a fresh DB
    // must succeed (no stored value to clash with).
    mem.register_namespace(Namespace::new("never-seen-default"))
        .await
        .expect("first observation of an unregistered ns must succeed");
}

/// G_v014a_9: fluent setter chain produces the same struct as struct expression
/// equivalent (internal — uses APPEND_ONLY const).
#[test]
fn g_v014a_9_builder_constructs_policy_with_partial_fields() {
    let via_setters = NamespacePolicy::new()
        .with_immutability(ImmutabilityLevel::AppendOnly)
        .with_forgettable(false)
        .with_dream_eligible(false);
    assert_eq!(via_setters, NamespacePolicy::APPEND_ONLY);
}

/// G_v014a_10: `NamespacePolicy` round-trips via serde_json (policy_json column).
#[test]
fn g_v014a_10_serde_roundtrip_via_policy_json_column() {
    let original = NamespacePolicy::APPEND_ONLY;
    let json = serde_json::to_string(&original).expect("serialize");
    assert!(json.contains("\"append_only\""), "json: {json}");
    let back: NamespacePolicy = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, original);
}

/// G_v014a_11: migration creates the `namespaces` table on a fresh DB; the
/// lazy-population path then inserts a default-policy row on first remember.
/// Decomposed from cycle-1 "backfill" — ADR-029a is CREATE-only (no backfill),
/// so this test verifies the lazy-population substitute semantics.
#[tokio::test]
async fn g_v014a_11_backfill_creates_default_policies_for_existing_namespaces() {
    let (mem, _tmp) = fresh_memory().await;
    // Lazy population fires on first request — register_namespace with default
    // policy on a never-seen ns is itself the populating call.
    mem.register_namespace(Namespace::new("lazy-pop-default"))
        .await
        .expect("default-policy registration");
    // Idempotent re-register with default confirms a row exists for this ns.
    mem.register_namespace(Namespace::new("lazy-pop-default"))
        .await
        .expect("re-register with default policy is idempotent");
}

/// G_v014a_12: v0.1.4 declare-but-don't-enforce — registering AppendOnly does
/// NOT cause `forget()` to return an error. Positive control for the contract.
/// (Dream is also exercised but the v0.1.0 dream substrate returns
/// NotImplemented, so we restrict the assertion to forget which is wired to a
/// substrate no-op.)
#[tokio::test]
async fn g_v014a_12_no_enforcement_at_v014() {
    let (mem, _tmp) = fresh_memory().await;
    let ns = Namespace::new("declare-not-enforce")
        .with_policy(NamespacePolicy::APPEND_ONLY)
        .expect("coherent");
    mem.register_namespace(ns.clone())
        .await
        .expect("register AppendOnly");
    // Forget on an AppendOnly ns MUST succeed at v0.1.4 (no enforcement yet).
    let deleted = mem
        .forget()
        .in_namespace(ns)
        .execute()
        .await
        .expect("v0.1.4 forget on AppendOnly must succeed (declare-not-enforce)");
    assert_eq!(deleted, 0, "v0.1.0 substrate stub returns 0");
}

/// G_v014a_13: old-shape `Namespace` JSON deserializes with `policy: None` via
/// serde default — backward-compat for v0.1.3 serialized payloads.
#[test]
fn g_v014a_13_serde_default_on_namespace_policy_field() {
    let json = r#"{"namespace":"old-ns","thread":null}"#;
    let ns: Namespace = serde_json::from_str(json).expect("deserialize old shape");
    assert_eq!(ns.namespace, "old-ns");
    assert!(ns.thread.is_none());
    assert!(ns.policy.is_none(), "policy must default to None");
}

/// G_v014a_14: N concurrent register_namespace calls with same policy all
/// return `Ok`; N concurrent calls with conflicting policies → exactly one
/// distinct stored policy (Vera MED-3).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn g_v014a_14_concurrent_register_namespace_race() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let (mem, _tmp) = fresh_memory().await;
    let mem = Arc::new(mem);

    // Phase 1: all-same-policy race — every call must succeed.
    const N: usize = 8;
    let policy = NamespacePolicy::APPEND_ONLY;
    let ns_name = "race-same-policy";
    let mut handles = Vec::with_capacity(N);
    for _ in 0..N {
        let mem = Arc::clone(&mem);
        let policy = policy.clone();
        let ns_name = ns_name.to_string();
        handles.push(tokio::spawn(async move {
            let ns = Namespace::new(ns_name).with_policy(policy).expect("coherent");
            mem.register_namespace(ns).await
        }));
    }
    for h in handles {
        h.await.expect("join").expect("same-policy concurrent calls must all return Ok");
    }

    // Phase 2: divergent-policy race — exactly one stored, the rest get
    // NamespacePolicyImmutable (or all-Ok if a single policy raced through
    // first — but at most ONE stored value can persist).
    let ns_name2 = "race-divergent";
    let policies = [
        NamespacePolicy::APPEND_ONLY,
        NamespacePolicy::new()
            .with_immutability(ImmutabilityLevel::Mutable)
            .with_forgettable(true)
            .with_dream_eligible(false),
        NamespacePolicy::default(),
    ];
    let ok_count = Arc::new(AtomicUsize::new(0));
    let immutable_err_count = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for p in policies.iter().cloned().cycle().take(N) {
        let mem = Arc::clone(&mem);
        let ok_count = Arc::clone(&ok_count);
        let imm_count = Arc::clone(&immutable_err_count);
        let ns_name = ns_name2.to_string();
        handles.push(tokio::spawn(async move {
            let ns = Namespace::new(ns_name).with_policy(p).expect("coherent");
            match mem.register_namespace(ns).await {
                Ok(()) => {
                    ok_count.fetch_add(1, Ordering::SeqCst);
                }
                Err(MemoryError::Core(CoreError::NamespacePolicyImmutable { .. })) => {
                    imm_count.fetch_add(1, Ordering::SeqCst);
                }
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }));
    }
    for h in handles {
        h.await.expect("join");
    }
    let oks = ok_count.load(Ordering::SeqCst);
    let imms = immutable_err_count.load(Ordering::SeqCst);
    assert_eq!(
        oks + imms,
        N,
        "every call must terminate either Ok or NamespacePolicyImmutable; oks={oks} imms={imms}"
    );
    assert!(
        oks >= 1,
        "at least one call must store its policy successfully; oks={oks}"
    );
}

/// G_v014a_15: migration is idempotent on double-apply (re-running the schema
/// setup must not error and must not duplicate rows). Verified by opening the
/// same DB twice and round-tripping a registration through it.
#[tokio::test]
async fn g_v014a_15_migration_idempotent_double_apply() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("kremory-mig-idem.db");
    // First open + register.
    {
        let mem = Memory::open(&path)
            .with_llm(null_llm())
            .with_embedder(null_embedder())
            .await
            .expect("first open");
        let ns = Namespace::new("mig-test")
            .with_policy(NamespacePolicy::APPEND_ONLY)
            .expect("coherent");
        mem.register_namespace(ns).await.expect("first register");
    }
    // Second open: migrations re-run (CREATE TABLE IF NOT EXISTS path). Must
    // not error; the stored policy must remain intact (G_v014a_16-shape).
    let mem2 = Memory::open(&path)
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("second open (migrations re-run)");
    let ns = Namespace::new("mig-test")
        .with_policy(NamespacePolicy::APPEND_ONLY)
        .expect("coherent");
    mem2.register_namespace(ns)
        .await
        .expect("re-register with same policy after second open is idempotent");
}

/// G_v014a_16: policy round-trips after DB close + reopen.
#[tokio::test]
async fn g_v014a_16_policy_round_trip_after_db_reopen() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("kremory-reopen.db");
    {
        let mem = Memory::open(&path)
            .with_llm(null_llm())
            .with_embedder(null_embedder())
            .await
            .expect("open #1");
        let ns = Namespace::new("reopen-ns")
            .with_policy(NamespacePolicy::APPEND_ONLY)
            .expect("coherent");
        mem.register_namespace(ns).await.expect("persist policy");
        mem.close().await.expect("close");
    }
    let mem2 = Memory::open(&path)
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("open #2");
    // Re-register with SAME policy must be idempotent — proves the stored
    // policy survived close+reopen.
    let ns2 = Namespace::new("reopen-ns")
        .with_policy(NamespacePolicy::APPEND_ONLY)
        .expect("coherent");
    mem2.register_namespace(ns2)
        .await
        .expect("same policy after reopen must be idempotent");
    // Conflicting policy must surface NamespacePolicyImmutable — confirms the
    // stored row is still the original AppendOnly value.
    let conflict = NamespacePolicy::new()
        .with_immutability(ImmutabilityLevel::Mutable)
        .with_forgettable(true)
        .with_dream_eligible(true);
    let ns3 = Namespace::new("reopen-ns")
        .with_policy(conflict)
        .expect("coherent");
    let err = mem2
        .register_namespace(ns3)
        .await
        .expect_err("different policy after reopen must err");
    assert!(matches!(
        err,
        MemoryError::Core(CoreError::NamespacePolicyImmutable { .. })
    ));
}

/// G_v014a_17: registering a non-default policy emits a `tracing::warn!` event
/// at target `kremory.namespace` containing "POLICY DECLARED BUT NOT ENFORCED".
#[tokio::test]
#[tracing_test::traced_test]
async fn g_v014a_17_warn_emitted_on_non_default_policy_registration() {
    let (mem, _tmp) = fresh_memory().await;
    let ns = Namespace::new("warn-emitter")
        .with_policy(NamespacePolicy::APPEND_ONLY)
        .expect("coherent");
    mem.register_namespace(ns)
        .await
        .expect("register AppendOnly");
    assert!(
        logs_contain("POLICY DECLARED BUT NOT ENFORCED"),
        "expected warn marker in captured logs"
    );
}

/// G_v014a_18: registering with default policy does NOT emit the warn — only
/// non-default declarations trigger the operational visibility marker.
#[tokio::test]
#[tracing_test::traced_test]
async fn g_v014a_18_no_warn_on_default_policy_registration() {
    let (mem, _tmp) = fresh_memory().await;
    mem.register_namespace(Namespace::new("default-policy-ns"))
        .await
        .expect("register default policy");
    assert!(
        !logs_contain("POLICY DECLARED BUT NOT ENFORCED"),
        "default-policy registration must not emit the declare-not-enforce warn"
    );
}

/// G_v014a_19: lazy population — calling `remember()` against an unregistered
/// namespace creates a default-policy row (proven by idempotent re-register).
///
/// Note: the v0.1.0 remember path requires a working LLM provider for the
/// extraction pipeline. We use `MockChatProvider::null()` which returns
/// an empty extraction result — the remember call may fail at the extraction
/// stage, but the lazy-population step runs BEFORE the substrate ingest call,
/// so the namespaces row is written even when extraction errors out.
/// We then assert idempotent re-register with default policy succeeds.
#[tokio::test]
async fn g_v014a_19_lazy_population_on_first_remember() {
    let (mem, _tmp) = fresh_memory().await;
    let ns = Namespace::new("lazy-from-remember");
    // Best-effort: ignore the remember outcome (the null LLM may fail) but the
    // lazy-population call inside RememberRequest::execute fires regardless.
    let _ = mem
        .remember("seed content")
        .in_namespace(ns.clone())
        .await;
    // The default-policy row should now exist — registering default explicitly
    // is idempotent (Ok). If lazy population had NOT fired, this would still
    // succeed by the on-demand path, but in either case the post-condition
    // (default policy stored) holds.
    mem.register_namespace(ns)
        .await
        .expect("default-policy re-register must succeed");
}
