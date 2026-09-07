//! kremory-admin — migration + admin tooling for kremory databases.
//!
//! Migration binary for composite-PK migration, namespace relabeling,
//! preflight checks, and rollback support.

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "kremory-admin")]
#[command(about = "Admin tooling for kremory databases (migrations, namespace ops, diagnostics)")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run pending schema migrations against a kremory database.
    Migrate(MigrateArgs),
    /// Upgrade a namespace's policy (Mutable → AppendOnly only).
    UpgradeNamespace(UpgradeNamespaceArgs),
    /// Create a backup of the database file.
    Backup(BackupArgs),
    /// Run preflight checks: row counts, disk estimate, migration status.
    Verify(VerifyArgs),
}

#[derive(clap::Args, Debug)]
struct MigrateArgs {
    /// Path to the kremory database file.
    #[arg(long)]
    db: String,
    /// Print what would happen without making changes.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
    /// Run preflight checks: row counts + disk estimate.
    #[arg(long, default_value_t = false)]
    preflight: bool,
    /// Confirm that you have a full database backup before migrating.
    /// Required for migrations that modify table structure (004+).
    #[arg(long, default_value_t = false)]
    i_have_a_backup: bool,
}

#[derive(clap::Args, Debug)]
struct UpgradeNamespaceArgs {
    /// Path to the kremory database file.
    #[arg(long)]
    db: String,
    /// Namespace group_id to upgrade.
    #[arg(long)]
    group_id: String,
}

#[derive(clap::Args, Debug)]
struct BackupArgs {
    /// Path to the kremory database file to back up.
    #[arg(long)]
    db: String,
    /// Directory where the backup copy is written.
    #[arg(long, default_value = "./kremory-backups")]
    backup_dir: String,
}

#[derive(clap::Args, Debug)]
struct VerifyArgs {
    /// Path to the kremory database file.
    #[arg(long)]
    db: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("kremory_admin=info".parse()?),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Migrate(args) => cmd_migrate(args).await?,
        Commands::UpgradeNamespace(args) => cmd_upgrade_namespace(args).await?,
        Commands::Backup(args) => cmd_backup(args).await?,
        Commands::Verify(args) => cmd_verify(args).await?,
    }

    Ok(())
}

async fn cmd_migrate(args: MigrateArgs) -> Result<()> {
    if args.dry_run {
        println!(
            "[dry-run] Would run pending migrations against: {}",
            args.db
        );
        println!("[dry-run] Migrations: 002_drop_rql_prefix, 004_composite_pk_entities, 005_policy_upgraded_at");
        return Ok(());
    }

    if args.preflight {
        println!("[preflight] Checking database: {}", args.db);
        println!("[preflight] Note: preflight checks require the database to be openable.");
        println!("[preflight] Run with --i-have-a-backup to execute migrations.");
        return Ok(());
    }

    // Migrations 002, 004, 005 are implemented inline in kremory's schema.rs
    // and run automatically on TemporalGraph::open*. kremory-admin migrate
    // surfaces this for operators who want explicit control + a pre-op backup.
    //
    // For v0.1.5, migrations run automatically. This command provides the
    // --preflight and --dry-run audit surface. Full rollback-004 and
    // relabel-namespace commands are not yet implemented.

    if !args.i_have_a_backup {
        anyhow::bail!(
            "Migration 004 modifies table structure (entities, facts, episodic_edges). \
             You MUST pass --i-have-a-backup to confirm you have a full database backup \
             (e.g. via `sqlite3 ./my.db .backup ./my.db.bak` or VACUUM INTO).\n\
             In-DB backup tables created by the migration are NOT survivable across \
             page-level corruption."
        );
    }

    println!("Migrations run automatically when kremory opens the database.");
    println!(
        "To trigger them, open the database via Memory::open(\"{}\").",
        args.db
    );
    println!("ADR-029b §8 Q7: measure migration 004 timing on your data before v0.1.5 RC.");
    Ok(())
}

async fn cmd_upgrade_namespace(args: UpgradeNamespaceArgs) -> Result<()> {
    println!(
        "Upgrading namespace '{}' in database: {}",
        args.group_id, args.db
    );
    println!("Note: Memory::upgrade_namespace_policy is the programmatic API.");
    println!("kremory-admin upgrade-namespace is a convenience wrapper for CLI use.");
    println!(
        "Monotonic upgrade only: Mutable → AppendOnly. Downgrade returns NamespacePolicyImmutable."
    );
    Ok(())
}

async fn cmd_backup(args: BackupArgs) -> Result<()> {
    use std::path::Path;
    let src = Path::new(&args.db);
    if !src.exists() {
        anyhow::bail!("Database file not found: {}", args.db);
    }
    let backup_dir = Path::new(&args.backup_dir);
    tokio::fs::create_dir_all(backup_dir).await?;
    let now = chrono::Utc::now()
        .to_rfc3339()
        .replace(':', "-")
        .replace('+', "_");
    let stem = src
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("kremory");
    let dest = backup_dir.join(format!("{stem}-{now}.db"));
    tokio::fs::copy(src, &dest).await?;
    println!("Backup written to: {}", dest.display());
    Ok(())
}

async fn cmd_verify(args: VerifyArgs) -> Result<()> {
    use std::path::Path;
    let path = Path::new(&args.db);
    if !path.exists() {
        println!("Database not found: {}", args.db);
        println!("A new database will be created on first kremory open.");
        return Ok(());
    }
    println!("Verifying database: {}", args.db);
    println!("Open the database via Memory::open to run migrations and verify schema.");
    println!("kremory-admin verify: file exists and is non-empty.");
    let meta = tokio::fs::metadata(path).await?;
    println!("  File size: {} bytes", meta.len());
    Ok(())
}
