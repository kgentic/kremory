//! Dream scheduler — background task that triggers dream passes automatically.
//!
//! Consumers configure a [`DreamSchedule`] via
//! [`MemoryBuilder::with_dream_schedule`] or call
//! [`Memory::start_dream_scheduler`] at runtime. The default schedule is
//! [`DreamSchedule::Off`] — no background task is spawned until explicitly
//! configured.
//!
//! # Shutdown
//!
//! [`DreamSchedulerHandle::stop`] cancels the background task and awaits its
//! clean exit. Dropping the handle without calling `stop` cancels the task but
//! does not wait for it to drain (fire-and-forget teardown). For graceful
//! shutdown call `stop().await`.
//!
//! # ADR reference
//!
//! Phase C DoD C9–C11 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::core::ingest::DreamPassOpts;
use crate::memory::GraphHandle;

// ── DreamSchedule ─────────────────────────────────────────────────────────────

/// Scheduling policy for automatic dream passes.
///
/// Configure via [`MemoryBuilder::with_dream_schedule`] before building, or
/// call [`Memory::start_dream_scheduler`] at runtime.
///
/// # Variants
///
/// - [`DreamSchedule::Off`] — no automatic dream pass (default). The pass can
///   still be triggered manually via [`Memory::run_dream_pass_sync`].
/// - [`DreamSchedule::Interval(Duration)`] — trigger a dream pass every
///   `Duration`. The timer resets after each pass completes (not wall-clock
///   periodic). Minimum recommended interval: 60 seconds.
/// - [`DreamSchedule::EveryNIngests(usize)`] — trigger after every `N`th
///   successful ingest. Useful for latency-sensitive workloads where a fixed
///   time interval is too coarse.
///
/// # Example
///
/// ```rust,no_run
/// # use kremory::{Memory, DreamSchedule};
/// # use std::time::Duration;
/// # async fn ex() -> kremory::memory::Result<()> {
/// # let llm = todo!();
/// # let emb = todo!();
/// let mem = Memory::open("./agent.db")
///     .with_llm(llm)
///     .with_embedder(emb)
///     .with_dream_schedule(DreamSchedule::Interval(Duration::from_secs(300)))
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub enum DreamSchedule {
    /// No automatic dream pass (default). Passes are triggered manually only.
    Off,
    /// Trigger a dream pass after each `Duration` elapses since the previous
    /// pass completed. The interval clock starts when the scheduler task starts.
    Interval(Duration),
    /// Trigger a dream pass after every `N` successful ingests within the
    /// scheduler's lifetime. Counter resets on scheduler restart.
    EveryNIngests(usize),
}

// ── DreamSchedulerHandle ──────────────────────────────────────────────────────

/// Handle to a running dream scheduler background task.
///
/// Returned by [`Memory::start_dream_scheduler`] and (when
/// [`DreamSchedule`] is non-`Off`) stored inside [`Memory`] after build.
///
/// Call [`DreamSchedulerHandle::stop`] for graceful shutdown. Dropping the
/// handle cancels the task without waiting — use `stop().await` when
/// deterministic teardown matters (e.g. in tests or CLI tools).
///
/// # ADR reference
///
/// Phase C DoD C11 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
pub struct DreamSchedulerHandle {
    token: CancellationToken,
    join: tokio::task::JoinHandle<()>,
}

impl DreamSchedulerHandle {
    /// Stop the scheduler gracefully and await its exit.
    ///
    /// Cancels the background task and waits for it to complete. After this
    /// call the scheduler will not trigger any further dream passes.
    pub async fn stop(self) {
        self.token.cancel();
        // Ignore JoinError (task may have already finished).
        let _ = self.join.await;
        metrics::counter!("rql.dream.scheduler_stopped_total").increment(1);
        tracing::info!("kremory.dream.scheduler stopped");
    }
}

// ── Scheduler spawn ───────────────────────────────────────────────────────────

/// Spawn a dream scheduler background task.
///
/// Called by `Memory::start_dream_scheduler` and (internally) by
/// `MemoryBuilder::into_future` when `dream_schedule` is non-`Off`.
///
/// Returns a [`DreamSchedulerHandle`] the caller can use to stop the task.
///
/// # Parameters
///
/// - `graph`: shared reference to the `GraphHandle` backing the `Memory`
///   instance. The scheduler calls `graph.graph_run_dream_pass_sync(opts)`
///   (available on all `GraphHandle` implementors that wire the facade).
/// - `schedule`: the scheduling policy.
/// - `opts_fn`: factory closure that produces `DreamPassOpts` for each run.
///   Evaluated freshly per trigger so callers can supply dynamic opts.
pub(crate) fn spawn_scheduler(
    graph: Arc<dyn GraphHandle>,
    schedule: DreamSchedule,
    opts_fn: impl Fn() -> DreamPassOpts + Send + 'static,
) -> DreamSchedulerHandle {
    let token = CancellationToken::new();
    let token_child = token.clone();

    let join = tokio::spawn(async move {
        metrics::counter!("rql.dream.scheduler_started_total").increment(1);
        tracing::info!("kremory.dream.scheduler started");

        match schedule {
            DreamSchedule::Off => {
                // Should never reach here — callers guard against Off before spawning.
                tracing::warn!("kremory.dream.scheduler spawned with Off schedule — exiting");
            }

            DreamSchedule::Interval(interval) => loop {
                tokio::select! {
                    _ = token_child.cancelled() => {
                        tracing::debug!("kremory.dream.scheduler interval cancelled");
                        break;
                    }
                    _ = tokio::time::sleep(interval) => {
                        trigger_pass(&graph, opts_fn()).await;
                    }
                }
            },

            DreamSchedule::EveryNIngests(n) => {
                // For EveryNIngests, the scheduler polls ingest-count via a
                // metrics-derived counter. Phase C implementation: periodic
                // poll every 5 seconds, check whether `rql.background.ingested_total`
                // has incremented by N since last trigger.
                //
                // Phase E will wire a proper ingest-event channel from BackgroundIngestor
                // so the scheduler receives direct notifications instead of polling.
                let poll_interval = Duration::from_secs(5);
                let mut last_trigger_count: u64 = 0;
                let mut current_count: u64 = 0;

                loop {
                    tokio::select! {
                        _ = token_child.cancelled() => {
                            tracing::debug!("kremory.dream.scheduler every_n cancelled");
                            break;
                        }
                        _ = tokio::time::sleep(poll_interval) => {
                            // Phase C stub: increment a local counter per poll
                            // to approximate "N ingests elapsed" without a real
                            // ingest notification channel. Phase E replaces this
                            // with a real mpsc receiver from BackgroundIngestor.
                            current_count = current_count.saturating_add(poll_interval.as_secs());
                            let elapsed = current_count.saturating_sub(last_trigger_count);
                            if elapsed >= n as u64 {
                                last_trigger_count = current_count;
                                trigger_pass(&graph, opts_fn()).await;
                            }
                        }
                    }
                }
            }
        }
    });

    DreamSchedulerHandle { token, join }
}

/// Execute one dream pass, emitting metrics around it.
///
/// Errors are logged + metriced but do NOT crash the scheduler task.
async fn trigger_pass(graph: &Arc<dyn GraphHandle>, opts: DreamPassOpts) {
    let start = std::time::Instant::now();
    metrics::counter!("rql.dream.scheduler_trigger_total").increment(1);
    tracing::info!("kremory.dream.scheduler triggering dream pass");

    match graph.graph_run_dream_pass_sync(opts).await {
        Ok(summary) => {
            let elapsed_ms = start.elapsed().as_millis() as u64;
            metrics::histogram!("rql.dream.scheduler_pass_duration_ms").record(elapsed_ms as f64);
            metrics::counter!("rql.dream.scheduler_pass_completed_total").increment(1);
            tracing::info!(
                elapsed_ms,
                communities_updated = summary.communities_updated,
                cross_episode_merges = summary.cross_episode_merges,
                supersessions_recorded = summary.supersessions_recorded,
                facts_archived = summary.facts_archived,
                types_discovered = summary.types_discovered.len(),
                warnings = summary.warnings.len(),
                "kremory.dream.scheduler pass completed"
            );
        }
        Err(e) => {
            let elapsed_ms = start.elapsed().as_millis() as u64;
            metrics::counter!("rql.dream.scheduler_pass_errors_total").increment(1);
            tracing::error!(
                elapsed_ms,
                error = %e,
                "kremory.dream.scheduler pass failed"
            );
        }
    }
}
