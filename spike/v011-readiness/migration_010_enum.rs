/// Spike 1: Migration 010 source-tier enum as TEXT CHECK constraint (libSQL in-memory)
///
/// Proves:
/// 1. CREATE TABLE with TEXT CHECK enumerating 5 ADR-045 §2 source-tier values compiles + runs
/// 2. All 5 valid values INSERT successfully
/// 3. An invalid value 'Garbage' is rejected by the CHECK constraint
/// 4. Re-running the same CREATE TABLE IF NOT EXISTS is a no-op (idempotent)
///
/// Run via: cargo run --bin migration_010_enum
///   (from spike/v011-readiness/)

use libsql::Builder;

#[tokio::main]
async fn main() {
    let mut all_pass = true;

    // --- Open in-memory DB (same pattern as TemporalGraph::open_in_memory) ---
    let db = Builder::new_local(":memory:")
        .build()
        .await
        .expect("build db");
    let conn = db.connect().expect("connect");

    // --- DDL: CREATE TABLE with TEXT CHECK constraint ---
    let create_sql = "
        CREATE TABLE IF NOT EXISTS entity_type_source_test (
            id          INTEGER PRIMARY KEY,
            label       TEXT    NOT NULL,
            source      TEXT    NOT NULL CHECK(source IN (
                            'Phase1Ner',
                            'Phase2Llm',
                            'DreamPass0',
                            'DreamPass1',
                            'ConsumerPinned'
                        )),
            assigned_at TEXT    NOT NULL,
            ner_conf    REAL
        )
    ";

    conn.execute(create_sql, ())
        .await
        .expect("first CREATE TABLE IF NOT EXISTS");

    // --- Check 1: idempotent re-run ---
    let idempotent = conn.execute(create_sql, ()).await.is_ok();
    print_check("Idempotent re-run (CREATE TABLE IF NOT EXISTS)", idempotent);
    if !idempotent { all_pass = false; }

    // --- Check 2: all 5 valid source-tier values insert ---
    let valid_values = [
        "Phase1Ner",
        "Phase2Llm",
        "DreamPass0",
        "DreamPass1",
        "ConsumerPinned",
    ];

    let mut all_valid_inserted = true;
    for (i, v) in valid_values.iter().enumerate() {
        let sql = format!(
            "INSERT INTO entity_type_source_test (label, source, assigned_at, ner_conf) \
             VALUES ('Entity{}', '{}', '2026-06-09T00:00:00Z', NULL)",
            i, v
        );
        match conn.execute(&sql, ()).await {
            Ok(_) => {}
            Err(e) => {
                eprintln!("  FAIL: INSERT '{}' rejected: {}", v, e);
                all_valid_inserted = false;
                all_pass = false;
            }
        }
    }
    print_check("All 5 valid source-tier values accepted", all_valid_inserted);

    // --- Check 3: invalid value 'Garbage' is rejected ---
    let bad_sql = "INSERT INTO entity_type_source_test (label, source, assigned_at, ner_conf) \
                   VALUES ('Bad', 'Garbage', '2026-06-09T00:00:00Z', NULL)";
    let invalid_rejected = conn.execute(bad_sql, ()).await.is_err();
    print_check("Invalid value 'Garbage' rejected by CHECK constraint", invalid_rejected);
    if !invalid_rejected { all_pass = false; }

    println!();
    if all_pass {
        println!("SPIKE 1: PASS");
    } else {
        println!("SPIKE 1: FAIL");
        std::process::exit(1);
    }
}

fn print_check(label: &str, pass: bool) {
    let mark = if pass { "PASS" } else { "FAIL" };
    println!("  [{}] {}", mark, label);
}
