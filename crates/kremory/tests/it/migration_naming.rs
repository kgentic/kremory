#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Story #34 — Migration naming: NNN_descriptive format enforcement.
//!
//! AC: every Migration::name must match `^\d{3}_[a-z][a-z0-9_]*$`.
//! validate_migration_set rejects non-conforming names.
//! Gates G2, G5.

/// Non-conforming name must be rejected by MigrationRunner.
///
/// This test drives through the public API surface (cargo test) to verify
/// that validate_migration_set rejects a name without the NNN_ prefix.
#[cfg(test)]
mod tests {
    use kremory::core::migrations::{Migration, MigrationRunner};

    async fn in_memory_conn() -> libsql::Connection {
        let db = libsql::Builder::new_local(":memory:")
            .build()
            .await
            .unwrap();
        let conn = db.connect().unwrap();
        conn.execute_batch(
            "CREATE TABLE app_meta (schema_version INTEGER DEFAULT 0);
             INSERT INTO app_meta VALUES (0);",
        )
        .await
        .unwrap();
        conn
    }

    #[tokio::test]
    async fn validate_rejects_non_conforming_migration_name() {
        let conn = in_memory_conn().await;
        // "create_foo" has no NNN_ prefix — must be rejected.
        let bad = [Migration {
            version: 1,
            name: "create_foo",
            sql: "CREATE TABLE IF NOT EXISTS foo (id INTEGER PRIMARY KEY)",
        }];
        let runner = MigrationRunner::new(&conn, "schema_version");
        let err = runner
            .run(&bad)
            .await
            .expect_err("must reject name without NNN_ prefix");
        let msg = format!("{err}");
        assert!(
            msg.contains("NNN_") || msg.contains("naming") || msg.contains("invalid"),
            "expected naming rejection error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn validate_accepts_conforming_migration_name() {
        let conn = in_memory_conn().await;
        // "001_create_foo" follows the NNN_descriptive convention — must pass.
        let good = [Migration {
            version: 1,
            name: "001_create_foo",
            sql: "CREATE TABLE IF NOT EXISTS foo (id INTEGER PRIMARY KEY)",
        }];
        let runner = MigrationRunner::new(&conn, "schema_version");
        runner
            .run(&good)
            .await
            .expect("conforming name must be accepted");
    }

    #[tokio::test]
    async fn validate_rejects_name_with_uppercase() {
        let conn = in_memory_conn().await;
        let bad = [Migration {
            version: 1,
            name: "001_CreateFoo",
            sql: "CREATE TABLE IF NOT EXISTS foo (id INTEGER PRIMARY KEY)",
        }];
        let runner = MigrationRunner::new(&conn, "schema_version");
        let err = runner
            .run(&bad)
            .await
            .expect_err("must reject uppercase in name");
        let msg = format!("{err}");
        assert!(
            msg.contains("NNN_") || msg.contains("naming") || msg.contains("invalid"),
            "expected naming rejection, got: {msg}"
        );
    }

    #[tokio::test]
    async fn validate_rejects_name_with_fewer_than_three_digits() {
        let conn = in_memory_conn().await;
        let bad = [Migration {
            version: 1,
            name: "01_create_foo",
            sql: "CREATE TABLE IF NOT EXISTS foo (id INTEGER PRIMARY KEY)",
        }];
        let runner = MigrationRunner::new(&conn, "schema_version");
        let err = runner
            .run(&bad)
            .await
            .expect_err("must reject fewer than 3-digit prefix");
        let msg = format!("{err}");
        assert!(
            msg.contains("NNN_") || msg.contains("naming") || msg.contains("invalid"),
            "expected naming rejection, got: {msg}"
        );
    }
}
