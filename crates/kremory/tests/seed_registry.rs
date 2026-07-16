#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Custom entity-type seed registry — facade-level integration tests
//! (custom-entity-type-registry spec §5.13, T5–T9 + T11).
//!
//! These exercise [`Memory::register_namespace_with_seed`] (the facade entry
//! point) over a real migrated `Memory` built through the builder path, asserting
//! the persisted `entity_types` table state. The apply-layer (`apply_namespace_seed`)
//! unit/in-process coverage lives in `src/core/entity_types.rs` tests; this file
//! proves the SAME contract holds through the public `register_namespace_with_seed`
//! surface AND that seed + namespace-policy land in ONE `BEGIN IMMEDIATE` txn
//! (T9 — rollback leaves zero rows + no policy).

use std::sync::Arc;

use kremory::{
    DynEmbeddingProvider, EntityTypeSpec, ImmutabilityLevel, Memory, NamespacePolicy,
    NamespaceRegistrationError, NamespaceSeed, SeedOutcome,
};
use tempfile::TempDir;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// Fresh temp dir + a `Memory` opened against a unique DB file inside it.
/// The `TempDir` is returned so the caller keeps it alive for the test scope.
async fn fresh_memory() -> (Memory, TempDir) {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("kremory-seed-registry.db");
    let mem = Memory::open(&path)
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("Memory::open builder");
    (mem, tmp)
}

fn spec(id: u32, name: &str) -> EntityTypeSpec {
    EntityTypeSpec {
        id,
        name: name.to_string(),
        description: format!("{name} description."),
    }
}

/// Read the `entity_types` rows for `group_id` directly from the persisted DB,
/// id-ascending, so tests can snapshot exact registry state.
async fn read_rows(mem: &Memory, group_id: &str) -> Vec<(u32, String)> {
    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path");
    let mut rows = tg
        .conn
        .query(
            "SELECT id, name FROM entity_types WHERE group_id = ?1 ORDER BY id ASC",
            libsql::params![group_id.to_string()],
        )
        .await
        .expect("entity_types query");
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.expect("row iter") {
        let id: i64 = row.get(0).expect("id col");
        let name: String = row.get(1).expect("name col");
        out.push((id as u32, name));
    }
    out
}

/// Count `entity_types` rows for `group_id`.
async fn count_rows(mem: &Memory, group_id: &str) -> usize {
    read_rows(mem, group_id).await.len()
}

/// Whether a `namespaces` policy row exists for `group_id` (used by the T9
/// rollback assertion: the policy write must be undone when the seed fails).
async fn policy_row_exists(mem: &Memory, group_id: &str) -> bool {
    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path");
    let mut rows = tg
        .conn
        .query(
            "SELECT COUNT(*) FROM namespaces WHERE group_id = ?1",
            libsql::params![group_id.to_string()],
        )
        .await
        .expect("namespaces count query");
    let row = rows.next().await.expect("row iter").expect("row");
    let n: i64 = row.get(0).expect("count col");
    n > 0
}

// ── T5–T8, T11: register_namespace_with_seed contract through the facade ─────

/// T5 — `Replace` on a FRESH namespace → `Seeded`; exactly the seeded rows
/// (+ injected catch-all) are persisted.
#[tokio::test]
async fn t5_facade_replace_fresh_namespace_seeded() {
    let (mem, _tmp) = fresh_memory().await;
    let ns = kremory::Namespace::new("legal");
    let outcome = mem
        .register_namespace_with_seed(
            ns,
            NamespaceSeed::Replace(vec![spec(11, "Court"), spec(12, "Judge")]),
        )
        .await
        .expect("Replace on a fresh namespace must seed");
    match outcome {
        SeedOutcome::Seeded { rows_written } => assert_eq!(rows_written, 3),
        other => panic!("expected Seeded, got {other:?}"),
    }
    let rows = read_rows(&mem, "legal").await;
    assert_eq!(
        rows,
        vec![
            (0, "Entity".to_string()),
            (11, "Court".to_string()),
            (12, "Judge".to_string())
        ],
        "catch-all + Court + Judge must be the exact persisted set"
    );
}

/// T6 (load-bearing, D9 / ASMP-003 regression) — `Replace` on a POPULATED
/// namespace with a DIFFERENT taxonomy → `Err(AlreadyPopulated)`, and the
/// persisted table is byte-identical before/after (no write, no orphaning).
#[tokio::test]
async fn t6_facade_replace_populated_divergent_fails_loud_no_write() {
    let (mem, _tmp) = fresh_memory().await;
    mem.register_namespace_with_seed(
        kremory::Namespace::new("legal"),
        NamespaceSeed::Replace(vec![spec(11, "Court"), spec(12, "Judge")]),
    )
    .await
    .expect("initial seed");
    let before = read_rows(&mem, "legal").await;

    let err = mem
        .register_namespace_with_seed(
            kremory::Namespace::new("legal"),
            NamespaceSeed::Replace(vec![spec(11, "Drug"), spec(12, "Condition")]),
        )
        .await
        .expect_err("divergent Replace on a populated namespace must fail loud");
    match err {
        NamespaceRegistrationError::AlreadyPopulated { group_id } => {
            assert_eq!(group_id, "legal");
        }
        other => panic!("expected AlreadyPopulated, got {other:?}"),
    }

    let after = read_rows(&mem, "legal").await;
    assert_eq!(
        before, after,
        "table must be byte-identical after a refused Replace (ASMP-003 no orphaning)"
    );
}

/// T7 — `Augment` on a fresh namespace → defaults (0–9) + custom (≥10) present.
///
/// Custom names MUST avoid the `DEFAULT_ENTITY_TYPES` vocabulary: a custom spec
/// whose name collides with a default (e.g. "Person") would be dropped by the
/// `UNIQUE(group_id, name)` constraint (`INSERT OR IGNORE`). "Statute" / "Judge"
/// are non-default. (Legal `Court` is now an opt-in augment, not a default —
/// see `td079_court_is_legal_augment_not_general_default`.)
#[tokio::test]
async fn t7_facade_augment_fresh_namespace_defaults_plus_custom() {
    let (mem, _tmp) = fresh_memory().await;
    mem.register_namespace_with_seed(
        kremory::Namespace::new("legal"),
        NamespaceSeed::Augment(vec![spec(11, "Statute"), spec(12, "Judge")]),
    )
    .await
    .expect("augment on fresh namespace");
    let rows = read_rows(&mem, "legal").await;
    // Catch-all default present.
    assert!(
        rows.iter().any(|(id, name)| *id == 0 && name == "Entity"),
        "default catch-all id=0 'Entity' must be present after Augment"
    );
    // Custom present.
    assert!(rows.iter().any(|(id, name)| *id == 11 && name == "Statute"));
    assert!(rows.iter().any(|(id, name)| *id == 12 && name == "Judge"));
    // Augment keeps the full default vocabulary, so the count exceeds the
    // 2 custom rows + catch-all (defaults occupy 0–9).
    assert!(
        rows.len() > 3,
        "Augment must retain the default vocabulary, got {} rows",
        rows.len()
    );
}

/// T8 — `AlreadySeeded` (not error) for `Default`/`Augment` on a populated ns,
/// with no table mutation.
#[tokio::test]
async fn t8_facade_default_augment_populated_already_seeded() {
    let (mem, _tmp) = fresh_memory().await;
    mem.register_namespace_with_seed(
        kremory::Namespace::new("legal"),
        NamespaceSeed::Replace(vec![spec(11, "Court")]),
    )
    .await
    .expect("initial seed");
    let snapshot = read_rows(&mem, "legal").await;

    let d = mem
        .register_namespace_with_seed(kremory::Namespace::new("legal"), NamespaceSeed::Default)
        .await
        .expect("Default on populated");
    assert_eq!(d, SeedOutcome::AlreadySeeded);

    let a = mem
        .register_namespace_with_seed(
            kremory::Namespace::new("legal"),
            NamespaceSeed::Augment(vec![spec(99, "Witness")]),
        )
        .await
        .expect("Augment on populated");
    assert_eq!(a, SeedOutcome::AlreadySeeded);

    assert_eq!(
        read_rows(&mem, "legal").await,
        snapshot,
        "Default/Augment on a populated namespace must not mutate the table"
    );
}

/// T9 (load-bearing) — `register_namespace_with_seed` lands seed + namespace
/// policy in ONE `BEGIN IMMEDIATE` txn. Injected failpoint: a divergent
/// `Replace` on a namespace that already has `entity_types` rows but NO policy
/// row (manufactured to mirror a pre-policy-era populated namespace). The call
/// writes the (non-default) policy first, then the seed fails with
/// `AlreadyPopulated`, triggering a rollback. Post-condition: ZERO new
/// entity_types rows AND the policy write is undone (no policy row) — proving
/// both writes were in the same transaction.
#[tokio::test]
async fn t9_facade_seed_and_policy_one_txn_rollback_leaves_zero_rows() {
    let (mem, _tmp) = fresh_memory().await;
    let group_id = "legal";

    // Manufacture a populated-but-policyless namespace: insert entity_types rows
    // directly (no namespace policy row written). This is the deterministic
    // failpoint — the seed step will refuse (divergent Replace on populated),
    // AFTER the policy gate has written a fresh policy row in the same txn.
    {
        let tg = mem
            .temporal_graph_for_test()
            .expect("temporal_graph must be set on the builder path");
        for (id, name) in [(0u32, "Entity"), (11, "Court"), (12, "Judge")] {
            tg.conn
                .execute(
                    "INSERT INTO entity_types (group_id, id, name, description) \
                     VALUES (?1, ?2, ?3, ?4)",
                    libsql::params![group_id, id as i64, name, format!("{name} desc")],
                )
                .await
                .expect("manual seed insert");
        }
    }
    assert!(
        !policy_row_exists(&mem, group_id).await,
        "precondition: no policy row yet"
    );
    let rows_before = count_rows(&mem, group_id).await;

    // Non-default policy so the policy gate performs a real WRITE inside the txn.
    let ns = kremory::Namespace::new(group_id)
        .with_policy(
            NamespacePolicy::new()
                .with_immutability(ImmutabilityLevel::AppendOnly)
                // AppendOnly is coherent only when forget + dream are both off
                // (validate() rejects AppendOnly + forgettable/dream_eligible).
                .with_forgettable(false)
                .with_dream_eligible(false),
        )
        .expect("AppendOnly is a coherent policy");

    let err = mem
        .register_namespace_with_seed(
            ns,
            // Divergent vs the manufactured rows → seed step fails AlreadyPopulated.
            NamespaceSeed::Replace(vec![spec(11, "Drug"), spec(12, "Condition")]),
        )
        .await
        .expect_err("divergent Replace must fail and roll the whole txn back");
    assert!(
        matches!(err, NamespaceRegistrationError::AlreadyPopulated { .. }),
        "expected AlreadyPopulated, got {err:?}"
    );

    // Atomicity proof: the seed write rolled back (zero new rows) AND the policy
    // write rolled back (no policy row). If they were in separate transactions
    // the policy row would survive.
    assert_eq!(
        count_rows(&mem, group_id).await,
        rows_before,
        "a refused seed must add zero entity_types rows (rollback)"
    );
    assert!(
        !policy_row_exists(&mem, group_id).await,
        "policy write must be rolled back with the failed seed (ONE txn) — \
         a surviving policy row proves the two writes were NOT atomic"
    );
}

/// T11 (load-bearing, D9a match-aware idempotency) — `Replace(seed)` on a
/// populated namespace whose rows EQUAL `seed` → `AlreadySeeded` (no write);
/// whose rows DIFFER → `Err(AlreadyPopulated)`. Proves the idempotent-boot
/// pattern is safe through the facade yet a genuine taxonomy change fails loud.
#[tokio::test]
async fn t11_facade_replace_match_aware_idempotency() {
    let (mem, _tmp) = fresh_memory().await;
    let seed = || NamespaceSeed::Replace(vec![spec(11, "Court"), spec(12, "Judge")]);

    // First boot: fresh → Seeded.
    let first = mem
        .register_namespace_with_seed(kremory::Namespace::new("legal"), seed())
        .await
        .expect("first boot seeds");
    assert!(matches!(first, SeedOutcome::Seeded { .. }));
    let snapshot = read_rows(&mem, "legal").await;

    // Second boot, IDENTICAL seed → AlreadySeeded, no write (D9a).
    let second = mem
        .register_namespace_with_seed(kremory::Namespace::new("legal"), seed())
        .await
        .expect("identical re-seed is idempotent");
    assert_eq!(
        second,
        SeedOutcome::AlreadySeeded,
        "identical Replace on every boot must be safe (D9a)"
    );
    assert_eq!(
        read_rows(&mem, "legal").await,
        snapshot,
        "identical re-seed must not mutate the table"
    );

    // Divergent seed → loud failure (D9).
    let err = mem
        .register_namespace_with_seed(
            kremory::Namespace::new("legal"),
            NamespaceSeed::Replace(vec![spec(11, "Court"), spec(12, "Magistrate")]),
        )
        .await
        .expect_err("a genuine taxonomy change must fail loud");
    assert!(
        matches!(err, NamespaceRegistrationError::AlreadyPopulated { .. }),
        "expected AlreadyPopulated, got {err:?}"
    );
}

/// TD-079 — `Court` is a legal-domain augment, NOT a universal default.
///
/// Substrate-purity guard: `DEFAULT_ENTITY_TYPES` must not leak legal
/// vocabulary into the general default set (a WTF for a first-time general
/// consumer). A legal consumer opts in via `NamespaceSeed::Augment(vec![Court])`;
/// a plain (non-augmented) default namespace never sees it.
#[tokio::test]
async fn td079_court_is_legal_augment_not_general_default() {
    let (mem, _tmp) = fresh_memory().await;

    // Legal namespace: augment the general defaults with the legal `Court` type
    // (id=10 — the first free slot above the 0–9 general defaults).
    mem.register_namespace_with_seed(
        kremory::Namespace::new("legal"),
        NamespaceSeed::Augment(vec![spec(10, "Court")]),
    )
    .await
    .expect("augment legal namespace with Court");

    // General namespace: the plain default vocabulary, no legal augmentation.
    mem.register_namespace_with_seed(kremory::Namespace::new("general"), NamespaceSeed::Default)
        .await
        .expect("seed general namespace with defaults");

    let legal_rows = read_rows(&mem, "legal").await;
    let general_rows = read_rows(&mem, "general").await;

    // Court IS available in the augmented legal namespace.
    assert!(
        legal_rows.iter().any(|(_, name)| name == "Court"),
        "Court must be present in the legal namespace after Augment; got {legal_rows:?}"
    );

    // Court is NOT present in the general default namespace (substrate-purity).
    assert!(
        !general_rows.iter().any(|(_, name)| name == "Court"),
        "Court must NOT leak into the general default namespace; got {general_rows:?}"
    );

    // The general namespace still carries the universal NER defaults.
    assert!(
        general_rows.iter().any(|(_, name)| name == "Person"),
        "general defaults must still include universal NER types like Person; got {general_rows:?}"
    );
}
