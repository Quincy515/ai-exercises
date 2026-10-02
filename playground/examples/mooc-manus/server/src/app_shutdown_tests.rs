//! 应用退出测试：验证有界收尾，以及真实 SSE 连接与任务注册表的关闭链路。

use crate::test_database as file_database;
#[path = "../tests/support/redis_database.rs"]
mod redis_database;

use std::{
    future::pending,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{ensure, Context, Result};
use async_trait::async_trait;
use axum::Router;
use loco_rs::{
    app::{AppContext, Hooks},
    config::Config,
    environment::Environment,
};
use migration::{Migrator, MigratorTrait, SchemaManager};
use serde_json::json;
use serial_test::serial;
use tokio::{
    sync::{oneshot, Notify},
    time::{timeout, Instant},
};

use super::{shutdown_agents, App};
use crate::{
    application::{services::AgentService, shutdown::ShutdownSignal},
    domain::{
        external::{SharedMessageQueue, SharedTask, Task, TaskRunner},
        models::{Event, Session, SessionStatus, TitleEvent},
        repositories::SessionRepository,
    },
    infrastructure::{external::RedisStreamTask, repositories::SeaOrmSessionRepository},
};

static DESTROY_CALLS: AtomicUsize = AtomicUsize::new(0);

struct ShutdownTask<const FAIL: bool>;

#[async_trait]
impl<const FAIL: bool> Task for ShutdownTask<FAIL> {
    async fn invoke(&self) -> Result<()> {
        unreachable!()
    }
    fn cancel(&self) -> bool {
        unreachable!()
    }
    fn input_stream(&self) -> SharedMessageQueue {
        unreachable!()
    }
    fn output_stream(&self) -> SharedMessageQueue {
        unreachable!()
    }
    fn id(&self) -> &str {
        unreachable!()
    }
    fn done(&self) -> bool {
        unreachable!()
    }
    fn get(_task_id: &str) -> Result<Option<SharedTask>> {
        unreachable!()
    }

    async fn destroy() -> Result<()> {
        DESTROY_CALLS.fetch_add(1, Ordering::SeqCst);
        ensure!(!FAIL, "任务销毁失败");
        Ok(())
    }
}

#[tokio::test]
#[serial(agent_shutdown)]
async fn service_shutdown_delegates_to_task_destroy_and_preserves_errors() {
    DESTROY_CALLS.store(0, Ordering::SeqCst);
    AgentService::shutdown::<ShutdownTask<false>>(&ShutdownSignal::default())
        .await
        .unwrap();
    let error = AgentService::shutdown::<ShutdownTask<true>>(&ShutdownSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "任务销毁失败");
    assert_eq!(DESTROY_CALLS.load(Ordering::SeqCst), 2);
}

#[tokio::test]
#[serial(agent_shutdown)]
async fn shutdown_waits_for_preparation_before_destroying_and_blocks_later_preparations() {
    DESTROY_CALLS.store(0, Ordering::SeqCst);
    let signal = ShutdownSignal::default();
    // 固定“聊天已进入准备、尚未完成注册”的窗口。
    let preparing = signal.begin_preparation().await.unwrap();
    let shutdown = AgentService::shutdown::<ShutdownTask<false>>(&signal);
    tokio::pin!(shutdown);
    assert!(futures::poll!(&mut shutdown).is_pending());
    signal.cancelled().await;
    assert_eq!(DESTROY_CALLS.load(Ordering::SeqCst), 0);

    // 准备完成或被取消后释放读锁，才允许销毁任务快照。
    drop(preparing);
    timeout(Duration::from_secs(1), shutdown)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(DESTROY_CALLS.load(Ordering::SeqCst), 1);
    assert!(signal.begin_preparation().await.is_none());
}

#[tokio::test(start_paused = true)]
async fn shutdown_completes_immediately_on_success_or_error() {
    let started = Instant::now();
    shutdown_agents(async { Ok(()) }).await;
    shutdown_agents(async { anyhow::bail!("测试清理失败") }).await;
    assert_eq!(Instant::now(), started);
}

#[tokio::test(start_paused = true)]
async fn shutdown_drops_unfinished_cleanup_after_thirty_seconds() {
    // Drop 哨兵证明超时释放等待中的 future；Tokio 虚拟时间避免真实等待30秒。
    struct Dropped(Arc<AtomicUsize>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let cleanup_drops = drops.clone();
    let started = Instant::now();
    shutdown_agents(async move {
        let _guard = Dropped(cleanup_drops);
        pending::<Result<()>>().await
    })
    .await;
    assert_eq!(started.elapsed(), Duration::from_secs(30));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[derive(Default)]
struct WaitingRunner {
    started: Notify,
    cancelled: AtomicUsize,
    completed: AtomicUsize,
    destroyed: AtomicUsize,
}

#[async_trait]
impl TaskRunner for WaitingRunner {
    async fn invoke(&self, _task: SharedTask) -> Result<()> {
        self.started.notify_one();
        pending().await
    }
    async fn on_cancel(&self, _task: SharedTask) -> Result<()> {
        self.cancelled.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn on_done(&self, _task: SharedTask) -> Result<()> {
        self.completed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn destroy(&self) -> Result<()> {
        self.destroyed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct RegisteredTask(RedisStreamTask);
impl Drop for RegisteredTask {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

struct RunningServer(tokio::task::JoinHandle<std::io::Result<()>>);
impl Drop for RunningServer {
    fn drop(&mut self) {
        self.0.abort();
    }
}

async fn test_context(
    database: &file_database::TestDatabase,
    redis_uri: &str,
) -> Result<AppContext> {
    let manager = SchemaManager::new(&database.db);
    let migrations = Migrator::migrations()
        .into_iter()
        .filter(|migration| {
            matches!(
                migration.name(),
                "m20260720_184611_sessions"
                    | "m20260720_191303_fix_sessions_table"
                    | "m20260526_131658_llm_configs"
                    | "m20260526_134746_fix_llm_configs_table"
                    | "m20260601_143631_agent_configs"
                    | "m20260601_144016_fix_agent_configs_table"
                    | "m20260605_185020_mcp_servers"
                    | "m20260717_191151_a2a_servers"
                    | "m20260718_113716_fix_a2a_servers_table"
            )
        })
        .collect::<Vec<_>>();
    ensure!(migrations.len() == 9, "没有找到会话与配置表的九条迁移");
    for migration in migrations {
        migration.up(&manager).await?;
    }
    let config: Config = serde_json::from_value(json!({
        "logger": {"enable": false, "level": "info", "format": "compact"},
        "server": {"port": 0, "host": "http://localhost"},
        "cache": {"kind": "Redis", "uri": redis_uri, "max_size": 1},
        "database": {"uri": "unused", "enable_logging": false,
            "min_connections": 1, "max_connections": 1,
            "connect_timeout": 10, "idle_timeout": 10}
    }))?;
    Ok(AppContext::builder(Environment::Test, database.db.clone(), config).build())
}

#[tokio::test]
#[serial(redis_stream_task)]
async fn app_shutdown_ends_live_sse_and_late_subscriptions_then_drains_server() -> Result<()> {
    let database = file_database::TestDatabase::new().await?;
    let redis = redis_database::TestRedis::new().await?;
    let ctx = test_context(&database, &redis.uri).await?;
    let router = App::routes(&ctx).to_router::<App>(ctx.clone(), Router::new())?;
    let repository = SeaOrmSessionRepository::new(database.db.clone());
    let runner = Arc::new(WaitingRunner::default());
    let task = RegisteredTask(RedisStreamTask::new(
        runner.clone(),
        redis.client.get_multiplexed_async_connection().await?,
    )?);
    let session = Session {
        status: SessionStatus::Running,
        task_id: Some(task.0.id().to_owned()),
        ..Session::default()
    };
    repository.save(session.clone()).await?;
    task.0.invoke().await?;
    timeout(Duration::from_secs(2), runner.started.notified()).await?;
    task.0
        .output_stream()
        .put(serde_json::to_value(Event::Title(TitleEvent {
            title: "退出测试".into(),
            ..TitleEvent::default()
        }))?)
        .await?;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let base_url = format!("http://{}", listener.local_addr()?);
    let (stop, stopping) = oneshot::channel();
    let (hook_done, hook_finished) = oneshot::channel();
    let (drain, draining) = oneshot::channel();
    // 与 Loco serve 相同：收到退出信号后执行 Hook，再由 Axum 等待活跃请求结束。
    // 门闩保留 Hook 完成到停止监听间的窗口，稳定验证晚到的订阅。
    let mut server = RunningServer(tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = stopping.await;
                App::on_shutdown(&ctx).await;
                let _ = hook_done.send(());
                let _ = draining.await;
            })
            .await
    }));
    let client = reqwest::Client::new();
    let mut list = timeout(
        Duration::from_secs(5),
        client
            .post(format!("{base_url}/api/sessions/stream"))
            .send(),
    )
    .await??;
    assert!(list.status().is_success());
    let list_chunk = timeout(Duration::from_secs(2), list.chunk())
        .await??
        .context("应先收到会话列表")?;
    assert!(std::str::from_utf8(&list_chunk)?.contains("event: sessions"));
    let mut chat = timeout(
        Duration::from_secs(5),
        client
            .post(format!("{base_url}/api/sessions/{}/chat", session.id))
            .json(&json!({}))
            .send(),
    )
    .await??;
    assert!(chat.status().is_success());
    let chat_chunk = timeout(Duration::from_secs(2), chat.chunk())
        .await??
        .context("应先收到已有任务事件")?;
    assert!(std::str::from_utf8(&chat_chunk)?.contains("event: title"));

    stop.send(()).unwrap();
    timeout(Duration::from_secs(3), hook_finished).await??;
    assert!(timeout(Duration::from_secs(2), list.bytes())
        .await??
        .is_empty());
    assert!(timeout(Duration::from_secs(2), chat.bytes())
        .await??
        .is_empty());
    assert_eq!(runner.cancelled.load(Ordering::SeqCst), 1);
    assert_eq!(runner.completed.load(Ordering::SeqCst), 1);
    assert_eq!(runner.destroyed.load(Ordering::SeqCst), 1);
    assert!(task.0.done());
    assert!(RedisStreamTask::get(task.0.id())?.is_none());

    // 已关闭状态保留在 watch 中；晚订阅立即结束，聊天流不会开始业务执行。
    let late_list = timeout(
        Duration::from_secs(2),
        client
            .post(format!("{base_url}/api/sessions/stream"))
            .send(),
    )
    .await??;
    assert!(late_list.status().is_success());
    assert!(timeout(Duration::from_secs(2), late_list.bytes())
        .await??
        .is_empty());
    // 使用不存在的会话：若执行了聊天业务，会产生 Error 事件；正确出口为空流。
    let late_chat = timeout(
        Duration::from_secs(2),
        client
            .post(format!(
                "{base_url}/api/sessions/{}/chat",
                uuid::Uuid::new_v4()
            ))
            .json(&json!({"message": "关闭后到达的新消息"}))
            .send(),
    )
    .await??;
    assert!(late_chat.status().is_success());
    assert!(timeout(Duration::from_secs(2), late_chat.bytes())
        .await??
        .is_empty());
    assert!(task.0.input_stream().is_empty().await?);
    assert!(repository
        .get_by_id(&session.id)
        .await?
        .unwrap()
        .events
        .is_empty());

    drain.send(()).unwrap();
    timeout(Duration::from_secs(3), &mut server.0)
        .await
        .context("SSE 已结束，Axum 应能完成优雅退出")???;
    Ok(())
}
