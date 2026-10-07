//! Startup schema migration must not queue behind busy tables.
//!
//! Requires `REMEMBRALL_SCHEMA_TEST_ADMIN_URL`: a connection URL for a role that
//! may CREATE/DROP DATABASE on a Postgres server with pgvector available (for
//! example `postgres://postgres:postgres@localhost:5450/postgres`). Every test
//! creates its own throwaway database named `remembrall_ddl_<pid>_<n>_test`,
//! works only inside it, and drops it afterwards. Never falls back to
//! DATABASE_URL or the application configuration, and never skips.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use remembrall_core::graph::store::GraphStore;
use remembrall_core::indexer::{Indexer, IndexerConfig};
use remembrall_core::memory::store::MemoryStore;
use remembrall_core::schema::{DDL_LOCK_TIMEOUT, SchemaReport};
use sqlx::postgres::PgPoolOptions;
use sqlx::{Connection, PgConnection, PgPool};

const SCHEMA: &str = "remembrall";
static NEXT_DB: AtomicU32 = AtomicU32::new(0);

struct TestDb {
    admin_url: String,
    name: String,
    url: String,
}

impl TestDb {
    async fn create() -> Self {
        let admin_url = std::env::var("REMEMBRALL_SCHEMA_TEST_ADMIN_URL").expect(
            "set REMEMBRALL_SCHEMA_TEST_ADMIN_URL to a server-admin URL (e.g. .../postgres); \
             tests create and drop their own throwaway databases",
        );
        let name = format!(
            "remembrall_ddl_{}_{}_test",
            std::process::id(),
            NEXT_DB.fetch_add(1, Ordering::SeqCst)
        );
        let mut admin = PgConnection::connect(&admin_url).await.unwrap();
        sqlx::query(&format!("CREATE DATABASE {name}"))
            .execute(&mut admin)
            .await
            .unwrap();
        admin.close().await.unwrap();
        let base = admin_url
            .rsplit_once('/')
            .map(|(base, _)| base)
            .expect("admin URL must name a database");
        let url = format!("{base}/{name}");
        Self {
            admin_url,
            name,
            url,
        }
    }

    async fn pool(&self) -> PgPool {
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .connect(&self.url)
            .await
            .unwrap();
        let current: String = sqlx::query_scalar("SELECT current_database()")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(current, self.name, "connected to the wrong database");
        pool
    }

    async fn drop_db(self) {
        let mut admin = PgConnection::connect(&self.admin_url).await.unwrap();
        sqlx::query(&format!(
            "DROP DATABASE IF EXISTS {} WITH (FORCE)",
            self.name
        ))
        .execute(&mut admin)
        .await
        .unwrap();
        admin.close().await.unwrap();
    }
}

/// Run `body` against a fresh throwaway database and always drop it afterwards.
async fn with_db<F, Fut>(body: F)
where
    F: FnOnce(String, PgPool) -> Fut,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let db = TestDb::create().await;
    let pool = db.pool().await;
    let outcome = tokio::spawn(body(db.url.clone(), pool.clone())).await;
    pool.close().await;
    db.drop_db().await;
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic.into_panic());
    }
}

/// The MCP server's startup path: memory store, then graph store.
async fn init_all(pool: &PgPool) -> SchemaReport {
    init_all_in(pool, SCHEMA).await
}

async fn init_all_in(pool: &PgPool, schema: &str) -> SchemaReport {
    let memory = MemoryStore::new(pool.clone(), schema.into()).unwrap();
    let graph = GraphStore::new(pool.clone(), schema.into()).unwrap();
    let mut report = memory.init_with_report().await.unwrap();
    report.extend(graph.init_with_report().await.unwrap());
    report
}

/// The indexer's startup path (graph store again, then file_index).
async fn init_indexer(pool: &PgPool) -> SchemaReport {
    let config = IndexerConfig {
        root_path: std::env::temp_dir(),
        project: "schema-test".into(),
        extensions: IndexerConfig::default_extensions(),
        ignore_patterns: IndexerConfig::default_ignore_patterns(),
    };
    Indexer::new(pool.clone(), SCHEMA.into(), config)
        .unwrap()
        .init_with_report()
        .await
        .unwrap()
}

async fn relation_exists(pool: &PgPool, name: &str) -> bool {
    sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace \
         WHERE n.nspname = $1 AND c.relname = $2)",
    )
    .bind(SCHEMA)
    .bind(name)
    .fetch_one(pool)
    .await
    .unwrap()
}

async fn column_exists(pool: &PgPool, table: &str, column: &str) -> bool {
    sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns \
         WHERE table_schema = $1 AND table_name = $2 AND column_name = $3)",
    )
    .bind(SCHEMA)
    .bind(table)
    .bind(column)
    .fetch_one(pool)
    .await
    .unwrap()
}

const RELATIONS: &[&str] = &[
    "memories",
    "idx_memories_type",
    "idx_memories_scope",
    "idx_memories_tags",
    "idx_memories_fingerprint",
    "idx_memories_embedding",
    "idx_memories_fts",
    "symbols",
    "relationships",
    "idx_symbols_file",
    "idx_symbols_name",
    "idx_symbols_parent",
    "idx_relationships_source",
    "idx_relationships_target",
    "file_index",
];

/// Open a second session that holds `lock_sql` inside an open transaction.
async fn hold_lock(url: &str, lock_sql: &str) -> PgConnection {
    let mut holder = PgConnection::connect(url).await.unwrap();
    sqlx::query("BEGIN").execute(&mut holder).await.unwrap();
    sqlx::query(lock_sql).execute(&mut holder).await.unwrap();
    holder
}

async fn release(mut holder: PgConnection) {
    sqlx::query("ROLLBACK").execute(&mut holder).await.unwrap();
    holder.close().await.unwrap();
}

#[tokio::test]
async fn fresh_database_gets_the_full_schema() {
    with_db(|_url, pool| async move {
        let report = init_all(&pool).await;
        let indexer = init_indexer(&pool).await;
        assert_eq!(indexer.applied, vec!["file_index table"], "{indexer:?}");
        assert!(
            report.deferred.is_empty(),
            "nothing to wait for: {report:?}"
        );
        // The memory store creates the shared schema, so only the graph store's
        // own CREATE SCHEMA is already satisfied; every other step runs.
        assert_eq!(report.skipped, vec!["schema"], "fresh database: {report:?}");
        assert_eq!(report.applied.len(), 17, "every other step ran: {report:?}");
        for name in RELATIONS {
            assert!(relation_exists(&pool, name).await, "{name} missing");
        }
        assert!(column_exists(&pool, "symbols", "parent_symbol_id").await);
        assert!(column_exists(&pool, "symbols", "moniker").await);
        let vector: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(vector);
    })
    .await;
}

/// A current schema must not request any lock on the tables: with every table
/// held ACCESS EXCLUSIVE by another session, even an ACCESS SHARE request would
/// block, so finishing at all proves no table lock was requested.
#[tokio::test]
async fn current_schema_executes_no_ddl_and_requests_no_table_lock() {
    with_db(|url, pool| async move {
        init_all(&pool).await;
        init_indexer(&pool).await;
        let holder = hold_lock(
            &url,
            "LOCK TABLE remembrall.memories, remembrall.symbols, remembrall.relationships, \
             remembrall.file_index IN ACCESS EXCLUSIVE MODE",
        )
        .await;

        let started = Instant::now();
        let report = tokio::time::timeout(Duration::from_secs(5), async {
            let mut report = init_all(&pool).await;
            report.extend(init_indexer(&pool).await);
            report
        })
        .await
        .expect("init on a current schema must not wait for table locks");
        let elapsed = started.elapsed();
        release(holder).await;

        assert!(report.applied.is_empty(), "no DDL may run: {report:?}");
        assert!(report.deferred.is_empty(), "nothing deferred: {report:?}");
        // 9 memory + 9 graph steps, then the indexer's 9 graph + 1 file_index.
        assert_eq!(report.skipped.len(), 28, "{report:?}");
        assert!(
            elapsed < DDL_LOCK_TIMEOUT,
            "took {elapsed:?}; a skipped step must not wait on a lock"
        );
    })
    .await;
}

/// Reproduces the field failure: a long `DELETE FROM symbols` holds ROW
/// EXCLUSIVE while the schema really is missing a column and an index. Startup
/// must give up on symbols after one lock timeout, keep going, and leave the
/// memory schema usable; the next start without the writer completes it.
#[tokio::test]
async fn outdated_schema_under_a_long_writer_defers_and_finishes_fast() {
    with_db(|url, pool| async move {
        init_all(&pool).await;
        sqlx::query("DROP INDEX remembrall.idx_symbols_name")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("ALTER TABLE remembrall.symbols DROP COLUMN moniker")
            .execute(&pool)
            .await
            .unwrap();

        let holder = hold_lock(
            &url,
            "DELETE FROM remembrall.symbols WHERE project = 'busy'",
        )
        .await;
        let started = Instant::now();
        let report = tokio::time::timeout(Duration::from_secs(10), init_all(&pool))
            .await
            .expect("init must not queue behind the writer");
        let elapsed = started.elapsed();

        // Memory tools are usable while the graph tables are busy.
        let memories: i64 = sqlx::query_scalar("SELECT count(*) FROM remembrall.memories")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(memories, 0);
        release(holder).await;

        assert!(
            report.deferred.contains(&"symbols field-tracking columns"),
            "{report:?}"
        );
        assert!(report.deferred.contains(&"idx_symbols_name"), "{report:?}");
        assert!(report.applied.is_empty(), "{report:?}");
        // One lock timeout for symbols, not one per statement.
        assert!(
            elapsed < DDL_LOCK_TIMEOUT * 2,
            "took {elapsed:?}; expected about one lock timeout"
        );
        assert!(!column_exists(&pool, "symbols", "moniker").await);

        let report = init_all(&pool).await;
        assert_eq!(
            report.applied,
            vec!["symbols field-tracking columns", "idx_symbols_name"],
            "{report:?}"
        );
        assert!(report.deferred.is_empty(), "{report:?}");
        assert!(column_exists(&pool, "symbols", "moniker").await);
        assert!(relation_exists(&pool, "idx_symbols_name").await);
    })
    .await;
}

/// When the relationships table is missing and cannot be created (its foreign
/// keys need a lock on the busy symbols table), the dependent indexes are
/// deferred rather than failing startup.
#[tokio::test]
async fn missing_table_defers_its_dependent_indexes() {
    with_db(|url, pool| async move {
        init_all(&pool).await;
        sqlx::query("DROP TABLE remembrall.relationships")
            .execute(&pool)
            .await
            .unwrap();
        let holder = hold_lock(
            &url,
            "DELETE FROM remembrall.symbols WHERE project = 'busy'",
        )
        .await;
        let report = tokio::time::timeout(Duration::from_secs(10), init_all(&pool))
            .await
            .expect("init must not queue behind the writer");
        release(holder).await;

        assert!(
            report.deferred.contains(&"relationships table"),
            "{report:?}"
        );
        assert!(
            report.deferred.contains(&"idx_relationships_source"),
            "{report:?}"
        );
        assert!(
            report.deferred.contains(&"idx_relationships_target"),
            "{report:?}"
        );

        let report = init_all(&pool).await;
        assert_eq!(
            report.applied,
            vec![
                "relationships table",
                "idx_relationships_source",
                "idx_relationships_target"
            ],
            "{report:?}"
        );
    })
    .await;
}

/// Unquoted DDL folds a mixed-case schema name to lower case; the catalog
/// probes must fold it the same way or nothing is ever recognised as present.
#[tokio::test]
async fn mixed_case_schema_name_is_probed_as_postgres_folds_it() {
    with_db(|_url, pool| async move {
        let report = init_all_in(&pool, "MixedCase").await;
        assert!(report.deferred.is_empty(), "{report:?}");
        assert_eq!(report.applied.len(), 17, "{report:?}");
        let report = init_all_in(&pool, "MixedCase").await;
        assert!(report.applied.is_empty(), "{report:?}");
        assert!(report.deferred.is_empty(), "{report:?}");
        assert_eq!(report.skipped.len(), 18, "{report:?}");
        let tables: i64 =
            sqlx::query_scalar("SELECT count(*) FROM pg_tables WHERE schemaname = 'mixedcase'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(tables, 3);
    })
    .await;
}
