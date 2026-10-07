//! The MCP server answers `initialize` promptly while the graph tables are busy.
//!
//! Field failure: a worktree daemon held a long `DELETE FROM remembrall.symbols`
//! and older servers queued `ALTER TABLE remembrall.symbols ...` behind it. Every
//! new server ran the same ALTER before answering `initialize`, joined the queue,
//! and the MCP client gave up after 30 s.
//!
//! Requires `REMEMBRALL_SCHEMA_TEST_ADMIN_URL`: a connection URL for a role that
//! may CREATE/DROP DATABASE on a Postgres server with pgvector available (for
//! example `postgres://postgres:postgres@localhost:5450/postgres`). The test
//! creates `remembrall_srv_<pid>_test`, runs only against it, and drops it.
//! The embedding model is loaded from `FASTEMBED_CACHE_DIR` when set, otherwise
//! from `$HOME/.fastembed_cache`.

use std::process::Stdio;
use std::time::{Duration, Instant};

use remembrall_core::graph::store::GraphStore;
use remembrall_core::memory::store::MemoryStore;
use sqlx::postgres::PgPoolOptions;
use sqlx::{Connection, PgConnection};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

/// Claude Code's MCP connect timeout is 30 s; leave a wide margin for model load.
const INITIALIZE_BUDGET: Duration = Duration::from_secs(15);

async fn admin() -> (String, PgConnection) {
    let admin_url = std::env::var("REMEMBRALL_SCHEMA_TEST_ADMIN_URL").expect(
        "set REMEMBRALL_SCHEMA_TEST_ADMIN_URL to a server-admin URL (e.g. .../postgres); \
         the test creates and drops its own throwaway database",
    );
    let conn = PgConnection::connect(&admin_url).await.unwrap();
    (admin_url, conn)
}

async fn run_scenario(url: String) -> Duration {
    // Current schema, as on any install that has started once.
    let pool = PgPoolOptions::new()
        .max_connections(2)
        .connect(&url)
        .await
        .unwrap();
    MemoryStore::new(pool.clone(), "remembrall".into())
        .unwrap()
        .init()
        .await
        .unwrap();
    GraphStore::new(pool.clone(), "remembrall".into())
        .unwrap()
        .init()
        .await
        .unwrap();
    pool.close().await;

    // A long writer on symbols ...
    let mut writer = PgConnection::connect(&url).await.unwrap();
    sqlx::query("BEGIN").execute(&mut writer).await.unwrap();
    sqlx::query("DELETE FROM remembrall.symbols WHERE project = 'busy'")
        .execute(&mut writer)
        .await
        .unwrap();

    // ... and an old-style server queued behind it for ACCESS EXCLUSIVE, which
    // makes every later request for any lock on symbols queue as well.
    let waiter_url = url.clone();
    let waiter = tokio::spawn(async move {
        let mut waiter = PgConnection::connect(&waiter_url).await.unwrap();
        let _ = sqlx::query(
            "ALTER TABLE remembrall.symbols \
             ADD COLUMN IF NOT EXISTS parent_symbol_id UUID, \
             ADD COLUMN IF NOT EXISTS moniker TEXT",
        )
        .execute(&mut waiter)
        .await;
    });
    let mut probe = PgConnection::connect(&url).await.unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let queued: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM pg_stat_activity \
             WHERE datname = current_database() AND wait_event_type = 'Lock' \
               AND query ILIKE 'ALTER TABLE remembrall.symbols%'",
        )
        .fetch_one(&mut probe)
        .await
        .unwrap();
        if queued >= 1 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the ALTER waiter never queued on symbols"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let cache = std::env::var("FASTEMBED_CACHE_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.fastembed_cache",
            std::env::var("HOME").expect("HOME or FASTEMBED_CACHE_DIR")
        )
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_remembrall"))
        .arg("serve")
        .env("REMEMBRALL_DATABASE_URL", &url)
        .env("DATABASE_URL", &url)
        .env("REMEMBRALL_SCHEMA", "remembrall")
        .env("FASTEMBED_CACHE_DIR", cache)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap()).lines();

    let started = Instant::now();
    stdin
        .write_all(
            br#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"startup-under-lock","version":"0"}}}
"#,
        )
        .await
        .unwrap();
    stdin.flush().await.unwrap();
    let line = tokio::time::timeout(INITIALIZE_BUDGET, stdout.next_line())
        .await
        .expect("server must answer initialize while symbols is busy")
        .unwrap()
        .expect("server closed stdout");
    let elapsed = started.elapsed();
    let response: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(response["id"], 1, "{line}");
    assert!(response["result"]["serverInfo"].is_object(), "{line}");

    stdin
        .write_all(
            br#"{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
"#,
        )
        .await
        .unwrap();
    stdin.flush().await.unwrap();
    let line = tokio::time::timeout(Duration::from_secs(5), stdout.next_line())
        .await
        .expect("tools/list must answer")
        .unwrap()
        .expect("server closed stdout");
    let response: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(response["id"], 2, "{line}");
    let tools = response["result"]["tools"].as_array().unwrap();
    assert!(
        tools.iter().any(|t| t["name"] == "remembrall_recall"),
        "{line}"
    );

    child.kill().await.unwrap();
    sqlx::query("ROLLBACK").execute(&mut writer).await.unwrap();
    writer.close().await.unwrap();
    waiter.await.unwrap();
    probe.close().await.unwrap();
    elapsed
}

#[tokio::test]
async fn initialize_answers_while_symbols_has_a_long_writer_and_a_queued_alter() {
    let (admin_url, mut admin) = admin().await;
    let name = format!("remembrall_srv_{}_test", std::process::id());
    sqlx::query(&format!("CREATE DATABASE {name}"))
        .execute(&mut admin)
        .await
        .unwrap();
    let base = admin_url
        .rsplit_once('/')
        .map(|(base, _)| base)
        .expect("admin URL must name a database");
    let url = format!("{base}/{name}");

    let outcome = tokio::spawn(run_scenario(url)).await;

    sqlx::query(&format!("DROP DATABASE IF EXISTS {name} WITH (FORCE)"))
        .execute(&mut admin)
        .await
        .unwrap();
    admin.close().await.unwrap();
    match outcome {
        Ok(elapsed) => eprintln!("initialize answered in {elapsed:?} under contention"),
        Err(panic) => std::panic::resume_unwind(panic.into_panic()),
    }
}
