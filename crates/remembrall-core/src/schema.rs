//! Lock-safe, idempotent startup schema migration.
//!
//! Postgres takes the table lock for `CREATE INDEX IF NOT EXISTS` (SHARE) and for
//! `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` (ACCESS EXCLUSIVE) *before* it checks
//! whether the index or column already exists. On a busy database a no-op
//! migration therefore queues behind any long writer of that table, and because
//! lock requests queue in arrival order it also blocks every later reader and
//! writer of the table. A server that runs that migration before answering the
//! MCP `initialize` request then misses the client's connect timeout.
//!
//! Every startup DDL statement is therefore paired with a catalog probe:
//!
//! 1. When the catalog already satisfies the statement it is skipped, and no lock
//!    is requested on the table at all.
//! 2. Otherwise the statement runs in its own transaction under a short
//!    `lock_timeout` ([`DDL_LOCK_TIMEOUT`]). A fresh database has nothing to wait
//!    for, so it gets exactly the statements it always got.
//! 3. If the statement fails, the probe is re-run (a concurrent starter may have
//!    created the object). A lock timeout that leaves the requirement unmet is
//!    logged as a warning and startup continues; any other error is returned.

use std::time::Duration;

use sqlx::PgPool;

use crate::error::{RemembrallError, Result};

/// How long a startup DDL statement may wait for a table lock before it is
/// deferred to a later start.
pub const DDL_LOCK_TIMEOUT: Duration = Duration::from_secs(2);

/// SQLSTATE `lock_not_available`, raised when `lock_timeout` expires.
const LOCK_NOT_AVAILABLE: &str = "55P03";

/// A catalog fact that makes a DDL statement a no-op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Requirement {
    /// `pg_extension` has the named extension.
    Extension(&'static str),
    /// The store's schema exists.
    Schema,
    /// A relation (table or index) with this name exists in the store's schema.
    Relation(&'static str),
    /// The table in the store's schema has this (non-dropped) column.
    Column {
        table: &'static str,
        column: &'static str,
    },
}

/// One idempotent startup DDL statement.
#[derive(Debug, Clone)]
pub struct DdlStep {
    /// Short name used in logs and in [`SchemaReport`].
    pub label: &'static str,
    /// Every requirement holds => the statement would be a no-op and is skipped.
    pub satisfies: Vec<Requirement>,
    /// Must hold before the statement can run (for example the table an index
    /// is built on). Unmet prerequisites defer the step instead of failing it.
    pub prerequisites: Vec<Requirement>,
    /// The existing table whose lock the statement needs, if any. Once one step
    /// times out on a table, later steps on the same table are deferred without
    /// waiting again, so a busy database costs at most one timeout per table.
    pub locks: Option<&'static str>,
    /// The statement itself (unchanged from the pre-guard migration).
    pub sql: String,
}

impl DdlStep {
    pub fn new(label: &'static str, satisfies: Vec<Requirement>, sql: String) -> Self {
        debug_assert!(
            !satisfies.is_empty(),
            "a step with no requirement would always be skipped"
        );
        Self {
            label,
            satisfies,
            prerequisites: Vec::new(),
            locks: None,
            sql,
        }
    }

    /// Declare the requirements that must hold before the statement can run.
    pub fn after(mut self, prerequisites: Vec<Requirement>) -> Self {
        self.prerequisites = prerequisites;
        self
    }

    /// Declare the existing table whose lock the statement takes.
    pub fn locking(mut self, table: &'static str) -> Self {
        self.locks = Some(table);
        self
    }
}

/// What happened to one [`DdlStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdlOutcome {
    /// The catalog already satisfied the step; nothing was executed.
    AlreadySatisfied,
    /// The statement was executed and committed.
    Applied,
    /// The statement could not take its lock in time, or a prerequisite is
    /// missing; it will be retried on the next start.
    Deferred,
}

/// Per-step outcomes of one schema initialization.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SchemaReport {
    pub applied: Vec<&'static str>,
    pub skipped: Vec<&'static str>,
    pub deferred: Vec<&'static str>,
}

impl SchemaReport {
    fn record(&mut self, label: &'static str, outcome: DdlOutcome) {
        match outcome {
            DdlOutcome::AlreadySatisfied => self.skipped.push(label),
            DdlOutcome::Applied => self.applied.push(label),
            DdlOutcome::Deferred => self.deferred.push(label),
        }
    }

    /// Fold another report into this one.
    pub fn extend(&mut self, other: SchemaReport) {
        self.applied.extend(other.applied);
        self.skipped.extend(other.skipped);
        self.deferred.extend(other.deferred);
    }
}

/// Check one requirement against the system catalogs. Catalog reads never take
/// a lock on the user table they describe.
pub async fn is_satisfied(pool: &PgPool, schema: &str, requirement: &Requirement) -> Result<bool> {
    // The DDL writes the schema name unquoted, so Postgres folds it to lower
    // case; `validate_schema_name` admits ASCII only, so this is the exact fold.
    let schema = schema.to_ascii_lowercase();
    let schema = schema.as_str();
    let exists: bool = match requirement {
        Requirement::Extension(name) => {
            sqlx::query_scalar(
                "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_extension WHERE extname = $1)",
            )
            .bind(name)
            .fetch_one(pool)
            .await?
        }
        Requirement::Schema => {
            sqlx::query_scalar(
                "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = $1)",
            )
            .bind(schema)
            .fetch_one(pool)
            .await?
        }
        Requirement::Relation(name) => {
            sqlx::query_scalar(
                "SELECT EXISTS (
                     SELECT 1 FROM pg_catalog.pg_class c
                     JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                     WHERE n.nspname = $1 AND c.relname = $2
                 )",
            )
            .bind(schema)
            .bind(name)
            .fetch_one(pool)
            .await?
        }
        Requirement::Column { table, column } => {
            sqlx::query_scalar(
                "SELECT EXISTS (
                     SELECT 1 FROM pg_catalog.pg_attribute a
                     JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
                     JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                     WHERE n.nspname = $1 AND c.relname = $2 AND a.attname = $3
                       AND a.attnum > 0 AND NOT a.attisdropped
                 )",
            )
            .bind(schema)
            .bind(table)
            .bind(column)
            .fetch_one(pool)
            .await?
        }
    };
    Ok(exists)
}

async fn all_satisfied(pool: &PgPool, schema: &str, requirements: &[Requirement]) -> Result<bool> {
    for requirement in requirements {
        if !is_satisfied(pool, schema, requirement).await? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn is_lock_timeout(error: &sqlx::Error) -> bool {
    matches!(
        error,
        sqlx::Error::Database(db) if db.code().as_deref() == Some(LOCK_NOT_AVAILABLE)
    )
}

async fn execute_with_lock_timeout(
    pool: &PgPool,
    sql: &str,
) -> std::result::Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;
    sqlx::query(&format!(
        "SET LOCAL lock_timeout = '{}ms'",
        DDL_LOCK_TIMEOUT.as_millis()
    ))
    .execute(&mut *tx)
    .await?;
    sqlx::query(sql).execute(&mut *tx).await?;
    tx.commit().await?;
    Ok(())
}

/// Run one step: skip it when the catalog already satisfies it, otherwise
/// execute it under [`DDL_LOCK_TIMEOUT`].
pub async fn ensure(pool: &PgPool, schema: &str, step: &DdlStep) -> Result<DdlOutcome> {
    if all_satisfied(pool, schema, &step.satisfies).await? {
        tracing::debug!(step = step.label, "schema: already current, skipped");
        return Ok(DdlOutcome::AlreadySatisfied);
    }
    if !all_satisfied(pool, schema, &step.prerequisites).await? {
        tracing::warn!(
            step = step.label,
            "schema: prerequisite missing (an earlier step was deferred); deferring to the next start"
        );
        return Ok(DdlOutcome::Deferred);
    }
    match execute_with_lock_timeout(pool, &step.sql).await {
        Ok(()) => {
            tracing::info!(step = step.label, "schema: applied");
            Ok(DdlOutcome::Applied)
        }
        Err(error) => {
            // A concurrent starter may have created the object while we waited.
            // If the re-check itself fails, report the original DDL error.
            let satisfied = match all_satisfied(pool, schema, &step.satisfies).await {
                Ok(satisfied) => satisfied,
                Err(probe_error) => {
                    tracing::warn!(step = step.label, %probe_error, "schema: re-check after a failed statement failed");
                    return Err(RemembrallError::Database(error));
                }
            };
            if satisfied {
                tracing::debug!(step = step.label, %error, "schema: satisfied concurrently");
                return Ok(DdlOutcome::AlreadySatisfied);
            }
            if is_lock_timeout(&error) {
                tracing::warn!(
                    step = step.label,
                    lock_timeout_ms = DDL_LOCK_TIMEOUT.as_millis() as u64,
                    "schema: table is busy (lock timeout); continuing without this change, it will be retried on the next start"
                );
                return Ok(DdlOutcome::Deferred);
            }
            Err(RemembrallError::Database(error))
        }
    }
}

/// Run every step in order and report what happened to each. A table that
/// timed out once is not waited on again by later steps in the same run.
pub async fn ensure_all(pool: &PgPool, schema: &str, steps: &[DdlStep]) -> Result<SchemaReport> {
    let mut report = SchemaReport::default();
    let mut busy: Vec<&'static str> = Vec::new();
    for step in steps {
        let outcome = match step.locks {
            Some(table) if busy.contains(&table) => {
                if all_satisfied(pool, schema, &step.satisfies).await? {
                    DdlOutcome::AlreadySatisfied
                } else {
                    tracing::warn!(
                        step = step.label,
                        table,
                        "schema: table already timed out this start; deferring to the next start"
                    );
                    DdlOutcome::Deferred
                }
            }
            _ => ensure(pool, schema, step).await?,
        };
        if outcome == DdlOutcome::Deferred {
            if let Some(table) = step.locks {
                if !busy.contains(&table) {
                    busy.push(table);
                }
            }
        }
        report.record(step.label, outcome);
    }
    if report.deferred.is_empty() {
        tracing::info!(
            applied = report.applied.len(),
            already_current = report.skipped.len(),
            "schema: startup migration complete"
        );
    } else {
        tracing::warn!(
            applied = report.applied.len(),
            already_current = report.skipped.len(),
            deferred = ?report.deferred,
            "schema: startup migration incomplete; deferred steps retry on the next start"
        );
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_records_each_outcome() {
        let mut report = SchemaReport::default();
        report.record("a", DdlOutcome::Applied);
        report.record("b", DdlOutcome::AlreadySatisfied);
        report.record("c", DdlOutcome::Deferred);
        let mut total = SchemaReport::default();
        total.extend(report);
        assert_eq!(total.applied, vec!["a"]);
        assert_eq!(total.skipped, vec!["b"]);
        assert_eq!(total.deferred, vec!["c"]);
    }
}
