//! ADR-066 dream CONSOLIDATION P2 — fact-archival deterministic-corpus harness.
//!
//! Spec `.ai-docs/specs/adr-066-dream-consolidation-impl-spec-2026-07-03.md` §3
//! (P2.1–P2.4) + §6 (archive corpus). The archive op is ZERO-LLM, so — like the P1
//! supersession harness and UNLIKE site3 — there is NO VCR cassette; each corpus row
//! is a fully-deterministic fixture. Mirrored from
//! `consolidation_supersession_test.rs`:
//!
//! - **per-row group isolation** — each row plants into its own fresh `group_id`;
//! - **smoke-one-before-batch** — `smoke_one_archive_harness` runs ONE representative
//!   row (`arc-001`, `long_expired_safe`) through the full pipeline FIRST;
//! - **o11y cross-check** — the pass's own
//!   `kremory.dream.consolidation.facts_archived_total` counter is snapshotted and
//!   asserted EQUAL to the harness-derived archive count (DoD-P2.3/P2.5).
//!
//! **Hard safety gate (spec §6):** ZERO false archives on ALL keep categories
//! (`within_grace`, `sole_binding`, `unexpired`), asserted EXACTLY (never `>=`). Plus,
//! per row: `facts` count decreases by `facts_archived`, `facts_archive` increases by
//! the same, and the archived fact's `facts_fts` shadow row is removed (RISK-003).

#![cfg(feature = "test-utils")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use chrono::{Duration, Utc};

use kremory::core::dream::archive;
use kremory::core::graph::InsertEntityWithGroupParams;
use kremory::core::schema::TemporalGraph;

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

const GRACE_DAYS: u32 = 90;

// ─── Corpus row ─────────────────────────────────────────────────────────────

/// Parsed corpus row. `expired_at_days_ago` is an offset from `now`; `null` =
/// `expired_at IS NULL` (unexpired → not a candidate). `sole_binding` rows plant NO
/// live anchor — the archived fact is the subject's ONLY reference, so the ref-count
/// guard (P2.2) must KEEP it. Non-`sole_binding` rows plant a live anchor so the
/// candidate is genuinely archivable.
struct Row {
    id: String,
    subject: String,
    predicate: String,
    object_value: String,
    category: String,
    valid_from_days_ago: i64,
    expired_at_days_ago: Option<i64>,
    sole_binding: bool,
    expected_archive: bool,
}

fn load_corpus() -> Vec<Row> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpora")
        .join("consolidation_archive_adversarial.jsonl");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read corpus at {path:?}: {e}"));

    let mut rows = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let raw: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("corpus line {line_no} failed to parse: {e}"));

        // Parse loudly — required fields carry NO default. A missing field is a corpus
        // authoring bug, surfaced as a panic (not a silent default).
        let str_field = |k: &str| -> String {
            raw[k]
                .as_str()
                .unwrap_or_else(|| panic!("corpus line {line_no}: missing str '{k}'"))
                .to_string()
        };
        let i64_field = |k: &str| -> i64 {
            raw[k]
                .as_i64()
                .unwrap_or_else(|| panic!("corpus line {line_no}: missing i64 '{k}'"))
        };
        let opt_i64_field = |k: &str| -> Option<i64> {
            match &raw[k] {
                serde_json::Value::Null => None,
                v => Some(
                    v.as_i64()
                        .unwrap_or_else(|| panic!("corpus line {line_no}: '{k}' not i64|null")),
                ),
            }
        };
        let bool_field = |k: &str| -> bool {
            raw[k]
                .as_bool()
                .unwrap_or_else(|| panic!("corpus line {line_no}: missing bool '{k}'"))
        };

        rows.push(Row {
            id: str_field("id"),
            subject: str_field("subject"),
            predicate: str_field("predicate"),
            object_value: str_field("object_value"),
            category: str_field("category"),
            valid_from_days_ago: i64_field("valid_from_days_ago"),
            expired_at_days_ago: opt_i64_field("expired_at_days_ago"),
            sole_binding: bool_field("sole_binding"),
            expected_archive: bool_field("expected_archive"),
        });
    }
    assert!(!rows.is_empty(), "corpus must not be empty");
    rows
}

// ─── Per-row isolated plant + run ───────────────────────────────────────────

/// Outcome of running one corpus row through the real archive sweep in its own group.
struct RowOutcome {
    row_id: String,
    category: String,
    expected_archive: bool,
    /// Did the sweep actually archive this fact?
    archived: bool,
    op_count: usize,
    /// `facts` row count for the group AFTER the sweep (candidate fact + optional
    /// anchor). Used to assert the delta == op_count.
    facts_before: i64,
    facts_after: i64,
    archive_rows_after: i64,
    fts_shadow_after: i64,
}

async fn insert_entity(graph: &TemporalGraph, gid: &str, id: &str) {
    let _ = graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some(gid),
        })
        .await;
}

/// Plant one row's candidate fact (+ a `facts_fts` shadow row, mirroring real ingest)
/// into a fresh isolated group and run `archive`. For non-`sole_binding` rows a live
/// anchor fact is planted so the candidate is genuinely archivable.
async fn run_one_row(row: &Row) -> RowOutcome {
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("open_in_memory");
    let gid = format!("arc-corpus-{}", row.id);
    let now = Utc::now();

    insert_entity(&graph, &gid, &row.subject).await;

    let valid_from = (now - Duration::days(row.valid_from_days_ago)).to_rfc3339();
    let expired_at = row
        .expired_at_days_ago
        .map(|d| (now - Duration::days(d)).to_rfc3339());
    let recorded_at = now.to_rfc3339();

    // Candidate fact.
    let candidate_id = insert_value_fact(
        &graph,
        &gid,
        &row.subject,
        &row.predicate,
        &row.object_value,
        &valid_from,
        expired_at.as_deref(),
        &recorded_at,
    )
    .await;
    // FTS shadow (real ingest inserts one for every object_value fact,
    // `facts.rs:370-377`).
    insert_fts_shadow(&graph, candidate_id, &row.predicate, &row.object_value).await;

    // Live anchor — only when NOT sole_binding. A sole_binding row deliberately has
    // no anchor so archiving its one fact would strand the subject (guard KEEPs it).
    if !row.sole_binding {
        let anchor_id = insert_value_fact(
            &graph,
            &gid,
            &row.subject,
            "anchor_pred",
            "anchor_val",
            &(now - Duration::days(1)).to_rfc3339(),
            None, // live (expired_at NULL)
            &recorded_at,
        )
        .await;
        insert_fts_shadow(&graph, anchor_id, "anchor_pred", "anchor_val").await;
    }

    let facts_before = count(
        &graph,
        "SELECT COUNT(*) FROM facts WHERE group_id = ?1",
        &gid,
    )
    .await;

    let report = archive(&graph, &gid, GRACE_DAYS).await.expect("archive");

    let facts_after = count(
        &graph,
        "SELECT COUNT(*) FROM facts WHERE group_id = ?1",
        &gid,
    )
    .await;
    let archive_rows_after = count(
        &graph,
        "SELECT COUNT(*) FROM facts_archive WHERE group_id = ?1",
        &gid,
    )
    .await;
    let fts_shadow_after = fts_shadow_count_for(&graph, candidate_id).await;

    RowOutcome {
        row_id: row.id.clone(),
        category: row.category.clone(),
        expected_archive: row.expected_archive,
        archived: report.count > 0,
        op_count: report.count,
        facts_before,
        facts_after,
        archive_rows_after,
        fts_shadow_after,
    }
}

#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
async fn insert_value_fact(
    graph: &TemporalGraph,
    gid: &str,
    subject: &str,
    predicate: &str,
    object_value: &str,
    valid_from: &str,
    expired_at: Option<&str>,
    recorded_at: &str,
) -> i64 {
    graph
        .conn
        .execute(
            "INSERT INTO facts \
             (subject_id, predicate, object_value, valid_from, recorded_at, \
              expired_at, group_id, subject_group_id, confidence) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1.0)",
            libsql::params![
                subject,
                predicate,
                object_value,
                valid_from,
                recorded_at,
                expired_at,
                gid,
                gid,
            ],
        )
        .await
        .expect("plant fact");
    let mut rows = graph
        .conn
        .query("SELECT last_insert_rowid()", ())
        .await
        .expect("rowid");
    rows.next()
        .await
        .expect("row")
        .expect("present")
        .get::<i64>(0)
        .expect("id")
}

#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
async fn insert_fts_shadow(
    graph: &TemporalGraph,
    fact_id: i64,
    predicate: &str,
    object_value: &str,
) {
    graph
        .conn
        .execute(
            "INSERT INTO facts_fts(fact_id, predicate, object_value) VALUES (?1, ?2, ?3)",
            libsql::params![fact_id, predicate, object_value],
        )
        .await
        .expect("plant fts shadow");
}

async fn fts_shadow_count_for(graph: &TemporalGraph, fact_id: i64) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM facts_fts WHERE fact_id = ?1",
            libsql::params![fact_id],
        )
        .await
        .expect("query facts_fts");
    rows.next()
        .await
        .expect("row")
        .expect("present")
        .get::<i64>(0)
        .expect("count")
}

async fn count(graph: &TemporalGraph, sql: &str, gid: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(sql, libsql::params![gid])
        .await
        .expect("count query");
    rows.next()
        .await
        .expect("row")
        .expect("present")
        .get::<i64>(0)
        .expect("count")
}

// ─── o11y cross-check ───────────────────────────────────────────────────────

/// Sum the archival counter across the snapshot.
fn sum_archived_counter(snapshotter: &Snapshotter) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != "kremory.dream.consolidation.facts_archived_total" {
                return None;
            }
            if let DebugValue::Counter(n) = value {
                Some(n)
            } else {
                None
            }
        })
        .sum()
}

// ─── Per-row invariant assertions (shared by smoke + batch) ─────────────────

/// Assert the structural invariants for one row's outcome (spec P2.2/P2.3/§6).
fn assert_row_invariants(o: &RowOutcome) {
    let keep_categories = ["within_grace", "sole_binding", "unexpired"];
    if keep_categories.contains(&o.category.as_str()) {
        // HARD safety gate: a keep-category fact must NEVER be archived (EXACT).
        assert!(
            !o.archived,
            "FALSE ARCHIVE: keep-category row {} ({}) was archived (op_count={})",
            o.row_id, o.category, o.op_count
        );
        assert_eq!(o.op_count, 0, "keep-category op_count must be EXACT 0");
        assert_eq!(
            o.facts_after, o.facts_before,
            "keep-category: live facts unchanged"
        );
        assert_eq!(
            o.archive_rows_after, 0,
            "keep-category: nothing in facts_archive"
        );
        assert_eq!(
            o.fts_shadow_after, 1,
            "keep-category: candidate's FTS shadow untouched"
        );
    } else {
        // long_expired_safe → archived exactly once.
        assert!(
            o.archived,
            "row {} ({}) should archive",
            o.row_id, o.category
        );
        assert_eq!(o.op_count, 1, "one fact archived for row {}", o.row_id);
        // facts decreased by op_count; facts_archive increased by the same.
        assert_eq!(
            o.facts_after,
            o.facts_before - o.op_count as i64,
            "live facts decreased by archived count (row {})",
            o.row_id
        );
        assert_eq!(
            o.archive_rows_after, o.op_count as i64,
            "facts_archive increased by archived count (row {})",
            o.row_id
        );
        // FTS shadow row removed for the archived id (RISK-003).
        assert_eq!(
            o.fts_shadow_after, 0,
            "archived fact's FTS shadow row removed (row {})",
            o.row_id
        );
    }
    // expected_archive from the corpus must match the observed outcome.
    assert_eq!(
        o.archived, o.expected_archive,
        "row {} ({}): observed archive != corpus expected_archive",
        o.row_id, o.category
    );
}

// ─── Smoke-one-before-batch (hard rule) ─────────────────────────────────────

#[tokio::test]
async fn smoke_one_archive_harness() {
    let corpus = load_corpus();
    let smoke = corpus
        .iter()
        .find(|r| r.id == "arc-001")
        .expect("corpus must contain arc-001");
    assert_eq!(smoke.category, "long_expired_safe");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let outcome = run_one_row(smoke).await;
    eprintln!(
        "[smoke-one] row_id={} category={} expected_archive={} archived={} op_count={} \
         facts {}->{} archive_rows={} fts_shadow_after={}",
        outcome.row_id,
        outcome.category,
        outcome.expected_archive,
        outcome.archived,
        outcome.op_count,
        outcome.facts_before,
        outcome.facts_after,
        outcome.archive_rows_after,
        outcome.fts_shadow_after,
    );

    assert_row_invariants(&outcome);

    // o11y cross-check on this single row.
    let counter = sum_archived_counter(&snapshotter);
    assert_eq!(
        counter, 1,
        "facts_archived counter must equal the single archive"
    );
}

// ─── Full batch: safety gate + delta invariants + o11y cross-check ──────────

#[tokio::test]
async fn archive_corpus_batch_metrics() {
    let corpus = load_corpus();

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mut outcomes = Vec::new();
    for row in &corpus {
        outcomes.push(run_one_row(row).await);
    }

    // Per-row structural invariants + hard safety gate.
    let mut false_archives = 0usize;
    let keep_categories = ["within_grace", "sole_binding", "unexpired"];
    for o in &outcomes {
        assert_row_invariants(o);
        if keep_categories.contains(&o.category.as_str()) && o.archived {
            false_archives += 1;
        }
    }

    // o11y cross-check (DoD-P2.3/P2.5): counter == sum of op counts == archived rows.
    let counter = sum_archived_counter(&snapshotter);
    let op_count_sum: usize = outcomes.iter().map(|o| o.op_count).sum();
    let archived_rows: usize = outcomes.iter().filter(|o| o.archived).count();

    eprintln!(
        "[batch] rows={} archived={archived_rows} op_count_sum={op_count_sum} \
         counter={counter} false_archives={false_archives}",
        outcomes.len(),
    );

    // Hard gate: ZERO false archives on keep categories (EXACT, never `>=`).
    assert_eq!(
        false_archives, 0,
        "hard gate: ZERO false archives on keep categories"
    );
    assert_eq!(
        counter as usize, op_count_sum,
        "facts_archived counter ({counter}) must equal summed op counts ({op_count_sum})"
    );
    assert_eq!(
        op_count_sum, archived_rows,
        "summed op counts must equal the number of archived facts"
    );
    // Sanity: the corpus DOES exercise the archive path (not a vacuous all-keep run).
    assert!(archived_rows > 0, "corpus must archive at least one fact");
}
