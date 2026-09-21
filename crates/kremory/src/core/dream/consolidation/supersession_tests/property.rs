use super::*;

// ══════════════════════════════════════════════════════════════════════════
// PROPERTY / INVARIANT TIER (INV1–INV8) — randomized-input safety proof.
//
// The op mutates USER facts, so its safety invariants must hold over RANDOMIZED
// inputs, not just hand-picked corpus rows. Each of N iterations builds an
// in-memory graph, plants K random facts across two namespaces, snapshots the
// facts table BEFORE, runs `supersession` on "gA" only, snapshots AFTER, and
// asserts eight invariants.
//
// PRNG choice: hand-rolled seeded SplitMix64 (NOT proptest's `proptest!` macro).
// Rationale — the op is `async` (proptest's macro drives sync closures; wrapping
// an async op per-case needs a runtime bridge that obscures the seed→case
// mapping the task demands). SplitMix64 is a tiny, well-known, statistically
// sound seedable generator. Seed = FIXED base ^ iteration index, so the whole
// test is deterministic + reproducible in CI (no wall-clock / random-device
// seeding). On any failure we print the exact `seed` + offending fact so the
// case reproduces from that one line.
// ══════════════════════════════════════════════════════════════════════════

/// Deterministic SplitMix64 PRNG (Vigna, public-domain reference). Seedable,
/// reproducible, no external dep. One `u64` of state; `next_u64` advances it.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, n)` (n > 0). Modulo bias is negligible for the tiny
    /// ranges used here (pools of ≤4, counts ≤30).
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// Uniform in `[lo, hi]` inclusive.
    fn in_range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.below(hi - lo + 1)
    }

    /// `true` with probability `num/den`.
    fn chance(&mut self, num: u64, den: u64) -> bool {
        self.below(den) < num
    }
}

/// A full snapshot row of a planted fact — every column the invariants read.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FactRow {
    id: i64,
    group_id: String,
    valid_to: Option<String>,
    expired_at: Option<String>,
    invalid_at: Option<String>,
    is_dream_generated: i64,
}

/// Snapshot the ENTIRE facts table (all groups), ordered by id, into `FactRow`s.
async fn snapshot_facts(graph: &TemporalGraph) -> Vec<FactRow> {
    let mut rows = graph
        .conn
        .query(
            "SELECT id, group_id, valid_to, expired_at, invalid_at, is_dream_generated \
             FROM facts ORDER BY id",
            (),
        )
        .await
        .expect("snapshot query");
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.expect("snapshot row") {
        out.push(FactRow {
            id: row.get(0).expect("id"),
            group_id: row.get(1).expect("group_id"),
            valid_to: row.get(2).expect("valid_to"),
            expired_at: row.get(3).expect("expired_at"),
            invalid_at: row.get(4).expect("invalid_at"),
            is_dream_generated: row.get(5).expect("is_dream_generated"),
        });
    }
    out
}

/// Would this row match the deterministic window-closeout predicate under `now`
/// for the swept group `"gA"`? (INV6's mechanical predicate.)
///
/// `valid_to NOT NULL AND valid_to < now AND expired_at NULL AND invalid_at NULL
///  AND is_dream_generated = 0 AND group_id = "gA"`.
/// String compare on RFC3339 UTC == chronological compare (op relies on this).
fn matches_closeout(row: &FactRow, now_rfc3339: &str) -> bool {
    row.group_id == "gA"
        && row.is_dream_generated == 0
        && row.expired_at.is_none()
        && row.invalid_at.is_none()
        && row.valid_to.as_deref().is_some_and(|vt| vt < now_rfc3339)
}

#[tokio::test]
async fn property_supersession_invariants_over_random_inputs() {
    // FIXED base seed — deterministic + reproducible (no wall-clock seeding).
    const BASE_SEED: u64 = 0x5150_5450_4159_4100; // "PPTPAYA\0"-ish, arbitrary fixed.
    const ITERATIONS: u64 = 300; // ≥ 200 required.

    // Small pools so entities repeat and namespaces collide meaningfully.
    const SUBJECTS: [&str; 4] = ["alice", "bob", "carol", "dave"];
    const PREDICATES: [&str; 4] = ["lives_in", "works_at", "role", "likes"];
    const GROUPS: [&str; 2] = ["gA", "gB"];

    for iter in 0..ITERATIONS {
        // Seed = base ^ iteration index → each case is independently reproducible.
        let seed = BASE_SEED ^ iter;
        let mut rng = SplitMix64::new(seed);

        let graph = TemporalGraph::open_in_memory()
            .await
            .unwrap_or_else(|e| panic!("seed={seed:#x}: open graph: {e}"));

        // `now` for THIS case, captured once so BEFORE/AFTER classification and
        // the op's own `Utc::now()` agree to within test wall-clock (facts are
        // planted at fixed offsets from this anchor, far from the boundary).
        let now = Utc::now();

        // Plant K random facts (K ∈ 3..=30).
        let k = rng.in_range(3, 30);
        for _ in 0..k {
            let base_subject = SUBJECTS[rng.below(SUBJECTS.len() as u64) as usize];
            let predicate = PREDICATES[rng.below(PREDICATES.len() as u64) as usize];
            let group = GROUPS[rng.below(GROUPS.len() as u64) as usize];
            // Namespace the subject by group so the SAME name never crosses
            // namespaces. `insert_entity_with_group` guards against a
            // cross-namespace name collision (bypass surface #2) and
            // returns an error the `ensure_entity` helper swallows — which would
            // then leave the fact's composite FK `(subject, group)` unsatisfied.
            // Subjects still REPEAT within a group (the small pool + the group
            // prefix), so entity reuse is exercised; only cross-group aliasing is
            // avoided (not the property under test here).
            let subject = format!("{group}_{base_subject}");
            let subject = subject.as_str();

            // valid_from: random PAST (1..=365 days ago).
            let valid_from = now - Duration::days(rng.in_range(1, 365) as i64);

            // valid_to ∈ {NULL, random-past, random-future} — weighted so all
            // three occur (≈ 1/3 each). Past window is CLOSED; future is OPEN.
            let valid_to = match rng.below(3) {
                0 => None,
                1 => Some(now - Duration::hours(rng.in_range(1, 240) as i64)), // past
                _ => Some(now + Duration::hours(rng.in_range(1, 240) as i64)), // future
            };

            // expired_at ∈ {NULL, random-past} — ~1/3 already handled.
            let expired_at = if rng.chance(1, 3) {
                Some(now - Duration::hours(rng.in_range(1, 480) as i64))
            } else {
                None
            };

            // invalid_at ∈ {NULL, random-past} — ~1/4 contradiction-handled.
            let invalid_at = if rng.chance(1, 4) {
                Some(now - Duration::hours(rng.in_range(1, 480) as i64))
            } else {
                None
            };

            // is_dream_generated ∈ {0, 1} — ~1/4 dream output (anti-loop input).
            let is_dream_generated = if rng.chance(1, 4) { 1 } else { 0 };

            let _id = plant_fact(
                &graph,
                group,
                subject,
                predicate,
                "val",
                valid_from,
                valid_to,
                expired_at,
                invalid_at,
                is_dream_generated,
            )
            .await;
        }

        // ── Snapshot BEFORE ──────────────────────────────────────────────────
        let before = snapshot_facts(&graph).await;

        // `now_rfc3339` used to classify BEFORE rows. Taken AFTER planting but
        // BEFORE the op runs; the op's own `Utc::now()` is a hair later, so any
        // row we classify as "past" (offset ≥ 1h from `now`) is unambiguously
        // past for the op too — no boundary flakiness.
        let now_rfc3339 = Utc::now().to_rfc3339();

        // ── Run the op on "gA" ONLY ──────────────────────────────────────────
        let report = supersession(SupersessionParams {
            graph: &graph,
            group_id: "gA",
            budget: &mut budget(),
            include_llm_nominate: false,
            model_id: "gemma4:e4b",
        })
        .await
        .unwrap_or_else(|e| panic!("seed={seed:#x}: supersession: {e}"));

        // ── Snapshot AFTER ───────────────────────────────────────────────────
        let after = snapshot_facts(&graph).await;

        // Index AFTER by id for O(1) lookup; op never inserts/deletes rows, so
        // BEFORE and AFTER have identical id sets.
        let after_by_id: std::collections::HashMap<i64, &FactRow> =
            after.iter().map(|r| (r.id, r)).collect();
        assert_eq!(
            before.len(),
            after.len(),
            "seed={seed:#x}: op must never insert or delete rows"
        );

        let mut expected_closeout_ids: Vec<i64> = Vec::new();

        for b in &before {
            let a = after_by_id
                .get(&b.id)
                .unwrap_or_else(|| panic!("seed={seed:#x}: fact {} vanished after op", b.id));

            let should_close = matches_closeout(b, &now_rfc3339);
            if should_close {
                expected_closeout_ids.push(b.id);
            }

            // A row is "unchanged" iff every column the op could touch is equal.
            // The op only ever writes `expired_at`; assert the FULL row is equal
            // for KEEP cases (stronger — proves nothing else drifted either).
            let unchanged = **a == *b;

            // INV1 (safety): valid_to IS NULL → UNCHANGED.
            if b.valid_to.is_none() {
                assert!(
                    unchanged,
                    "seed={seed:#x} INV1 violated: open-ended fact changed.\n  before={b:?}\n  after={a:?}"
                );
            }

            // INV2 (safety): valid_to >= now → UNCHANGED (future / equal window).
            if let Some(vt) = b.valid_to.as_deref() {
                if vt >= now_rfc3339.as_str() {
                    assert!(
                        unchanged,
                        "seed={seed:#x} INV2 violated: still-valid (valid_to>=now) fact changed.\n  before={b:?}\n  after={a:?}"
                    );
                }
            }

            // INV3 (no double-handle): already had expired_at OR invalid_at set
            // → UNCHANGED.
            if b.expired_at.is_some() || b.invalid_at.is_some() {
                assert!(
                    unchanged,
                    "seed={seed:#x} INV3 violated: already-resolved fact changed.\n  before={b:?}\n  after={a:?}"
                );
            }

            // INV4 (namespace isolation): group "gB" → UNCHANGED.
            if b.group_id == "gB" {
                assert!(
                    unchanged,
                    "seed={seed:#x} INV4 violated: other-namespace (gB) fact changed.\n  before={b:?}\n  after={a:?}"
                );
            }

            // INV5 (anti-loop): is_dream_generated = 1 → UNCHANGED.
            if b.is_dream_generated == 1 {
                assert!(
                    unchanged,
                    "seed={seed:#x} INV5 violated: dream-generated fact changed.\n  before={b:?}\n  after={a:?}"
                );
            }

            // INV6 (correctness): a matching gA fact must have expired_at == its
            // valid_to AFTER, and NOTHING else changed.
            if should_close {
                // expired_at now set to exactly valid_to.
                assert_eq!(
                    a.expired_at, b.valid_to,
                    "seed={seed:#x} INV6 violated: retired fact's expired_at != its valid_to.\n  before={b:?}\n  after={a:?}"
                );
                // Nothing else changed: every OTHER column identical.
                assert_eq!(a.group_id, b.group_id, "seed={seed:#x} INV6: group changed");
                assert_eq!(
                    a.valid_to, b.valid_to,
                    "seed={seed:#x} INV6: valid_to changed"
                );
                assert_eq!(
                    a.invalid_at, b.invalid_at,
                    "seed={seed:#x} INV6: invalid_at changed"
                );
                assert_eq!(
                    a.is_dream_generated, b.is_dream_generated,
                    "seed={seed:#x} INV6: is_dream_generated changed"
                );
            } else {
                // The COMPLEMENT of INV6: any row that does NOT match the
                // predicate must be fully unchanged (covers all KEEP paths at
                // once — INV1–INV5 are the named sub-cases of this).
                assert!(
                    unchanged,
                    "seed={seed:#x} INV6-complement violated: non-matching fact changed.\n  before={b:?}\n  after={a:?}"
                );
            }
        }

        // INV7 (count): report.count == number of rows matching the predicate.
        assert_eq!(
            report.count,
            expected_closeout_ids.len(),
            "seed={seed:#x} INV7 violated: report.count ({}) != matched-predicate count ({})",
            report.count,
            expected_closeout_ids.len()
        );

        // INV8 (idempotent): a SECOND run on "gA" retires 0 and changes nothing.
        let after_first = snapshot_facts(&graph).await;
        let second = supersession(SupersessionParams {
            graph: &graph,
            group_id: "gA",
            budget: &mut budget(),
            include_llm_nominate: false,
            model_id: "gemma4:e4b",
        })
        .await
        .unwrap_or_else(|e| panic!("seed={seed:#x}: second supersession: {e}"));
        let after_second = snapshot_facts(&graph).await;
        assert_eq!(
            second.count, 0,
            "seed={seed:#x} INV8 violated: second run retired {} (must be 0)",
            second.count
        );
        assert_eq!(
            after_first, after_second,
            "seed={seed:#x} INV8 violated: second run mutated the table"
        );
    }
}
