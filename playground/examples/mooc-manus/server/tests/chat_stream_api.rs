//! 通过真实路由、私有 PostgreSQL 与 Redis 验证聊天 SSE；任务替身覆盖运行与持久化。

#[path = "support/file_database.rs"]
mod file_database;
#[path = "support/redis_database.rs"]
mod redis_database;

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{ensure, Context, Result};
use async_trait::async_trait;
use axum::Router;
use axum_test::TestServer;
use chrono::{TimeZone, Utc};
use loco_rs::{
    app::{AppContext, Hooks},
    config::Config,
    environment::Environment,
};
use migration::{Migrator, MigratorTrait, SchemaManager};
use serde_json::{json, Value};
use serial_test::serial;
use server::{
    app::App,
    domain::{
        external::{SharedTask, Task, TaskRunner},
        models::{
            DoneEvent, Event, ExecutionStatus, MessageEvent, MessageRole, Plan, PlanEvent, Session,
            SessionStatus, ShellToolContent, Step, StepEvent, TitleEvent, ToolContent, ToolEvent,
            ToolEventStatus, ToolResult,
        },
        repositories::SessionRepository,
    },
    infrastructure::{external::RedisStreamTask, repositories::SeaOrmSessionRepository},
    views::events::AgentSseEvent,
};
use tokio::time::timeout;

const MESSAGE: &str = "整理本节资料";
// 0 是合法的 Unix 秒时间戳，不能被当作缺省值忽略。
const TIMESTAMP: i64 = 0;

struct TestApp {
    server: TestServer,
    database: file_database::TestDatabase,
    redis: redis_database::TestRedis,
}

impl TestApp {
    async fn new() -> Result<Self> {
        // 首次 HTTP 客户端会初始化系统 TLS；在夹具阶段完成，保留聊天事件流的五秒预算。
        drop(reqwest::Client::builder().build()?);
        let database = file_database::TestDatabase::new().await?;
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
        let redis = redis_database::TestRedis::new().await?;
        let config: Config = serde_json::from_value(json!({
            "logger": { "enable": false, "level": "info", "format": "compact" },
            "server": { "port": 0, "host": "http://localhost" },
            "cache": { "kind": "Redis", "uri": redis.uri, "max_size": 1 },
            "database": {
                "uri": "unused", "enable_logging": false,
                "min_connections": 1, "max_connections": 1,
                "connect_timeout": 10, "idle_timeout": 10
            }
        }))?;
        let ctx = AppContext::builder(Environment::Test, database.db.clone(), config).build();
        let router = App::routes(&ctx).to_router::<App>(ctx, Router::new())?;
        Ok(Self {
            server: TestServer::new(router)?,
            database,
            redis,
        })
    }

    fn repository(&self) -> SeaOrmSessionRepository {
        SeaOrmSessionRepository::new(self.database.db.clone())
    }
}

struct EchoRunner {
    repository: Arc<SeaOrmSessionRepository>,
    session_id: String,
    attachment_id: String,
    input_was_persisted: AtomicBool,
}

#[async_trait]
impl TaskRunner for EchoRunner {
    async fn invoke(&self, task: SharedTask) -> Result<()> {
        let (queue_id, payload) = task.input_stream().pop().await?.context("输入流缺少消息")?;
        let mut event: Event = serde_json::from_value(payload)?;
        event.set_id(queue_id);
        let session = self.repository.get_by_id(&self.session_id).await?.unwrap();
        ensure!(
            session.events == [event.clone()],
            "启动任务前必须持久化用户事件"
        );
        ensure!(session.latest_message == MESSAGE);
        ensure!(session.latest_message_at == Utc.timestamp_opt(TIMESTAMP, 0).single());
        let Event::Message(message) = event else {
            anyhow::bail!("输入流应包含消息事件");
        };
        ensure!(message.role == MessageRole::User && message.message == MESSAGE);
        ensure!(message.attachments.len() == 1);
        ensure!(message.attachments[0].id == self.attachment_id);
        self.input_was_persisted.store(true, Ordering::SeqCst);

        let reply = MessageEvent {
            role: MessageRole::Assistant,
            message: "资料已整理".to_owned(),
            ..MessageEvent::default()
        };
        self.repository
            .update_latest_message(&self.session_id, &reply.message, reply.base.created_at)
            .await?;
        self.repository
            .increment_unread_message_count(&self.session_id)
            .await?;
        for mut event in [Event::Message(reply), Event::Done(DoneEvent::default())] {
            let id = task
                .output_stream()
                .put(serde_json::to_value(&event)?)
                .await?;
            event.set_id(id);
            self.repository.add_event(&self.session_id, event).await?;
        }
        self.repository
            .update_status(&self.session_id, SessionStatus::Completed)
            .await
    }

    async fn destroy(&self) -> Result<()> {
        Ok(())
    }

    async fn on_done(&self, _task: SharedTask) -> Result<()> {
        Ok(())
    }
}

/// 清理只针对本测试注册的任务；已完成任务的监督协程会自行移除注册项。
struct RegisteredTask(Arc<RedisStreamTask>);

impl Drop for RegisteredTask {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

fn sse_events(body: &str) -> Result<Vec<(String, Value)>> {
    body.split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .map(|frame| {
            let event = frame
                .lines()
                .find_map(|line| line.strip_prefix("event:"))
                .context("SSE 缺少事件类型")?
                .trim()
                .to_owned();
            let data = frame
                .lines()
                .find_map(|line| line.strip_prefix("data:"))
                .context("SSE 缺少数据")?;
            Ok((event, serde_json::from_str(data)?))
        })
        .collect()
}

#[tokio::test]
#[serial]
async fn chat_stream_runs_a_real_registered_task_and_persists_queue_ids() -> Result<()> {
    let app = TestApp::new().await?;
    let repository = Arc::new(app.repository());
    let mut session = Session {
        status: SessionStatus::Running,
        unread_message_count: 4,
        ..Session::default()
    };
    let runner = Arc::new(EchoRunner {
        repository: repository.clone(),
        session_id: session.id.clone(),
        attachment_id: uuid::Uuid::new_v4().to_string(),
        input_was_persisted: AtomicBool::new(false),
    });
    let task = RegisteredTask(Arc::new(RedisStreamTask::new(
        runner.clone(),
        app.redis.client.get_multiplexed_async_connection().await?,
    )?));
    session.task_id = Some(task.0.id().to_owned());
    repository.save(session.clone()).await?;

    let request = json!({
        "message": MESSAGE,
        "attachments": [runner.attachment_id],
        "timestamp": TIMESTAMP
    });
    let response = timeout(Duration::from_secs(5), async {
        app.server
            .post(&format!("/api/sessions/{}/chat", session.id))
            .json(&request)
            .await
    })
    .await
    .context("聊天 SSE 应在任务输出 Done 后结束")?;
    response.assert_status_ok();
    response.assert_header("content-type", "text/event-stream");
    let events = sse_events(&response.text())?;
    timeout(Duration::from_secs(2), async {
        while !task.0.done() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await?;

    assert!(runner.input_was_persisted.load(Ordering::SeqCst));
    assert_eq!(
        events
            .iter()
            .map(|(kind, _)| kind.as_str())
            .collect::<Vec<_>>(),
        ["message", "done"]
    );
    let stored = repository.get_by_id(&session.id).await?.unwrap();
    assert_eq!(stored.events.len(), 3);
    for ((_, payload), stored_event) in events.iter().zip(&stored.events[1..]) {
        let response = serde_json::to_value(AgentSseEvent::from(stored_event.clone()))?;
        assert_eq!(payload, &response["data"]);
        assert!(payload.get("type").is_none() && payload.get("id").is_none());
        assert!(payload["created_at"].is_i64());
        let queue_id = payload["event_id"].as_str().unwrap();
        let (milliseconds, sequence) = queue_id.split_once('-').unwrap();
        milliseconds.parse::<u64>()?;
        sequence.parse::<u64>()?;
    }
    assert_eq!(stored.status, SessionStatus::Completed);
    assert_eq!(stored.unread_message_count, 0);
    assert_eq!(stored.latest_message, "资料已整理");
    assert!(task.0.input_stream().is_empty().await?);
    Ok(())
}

#[tokio::test]
#[serial]
async fn chat_stream_drains_queued_events_even_when_task_is_already_done() -> Result<()> {
    let app = TestApp::new().await?;
    let repository = Arc::new(app.repository());
    let mut session = Session {
        status: SessionStatus::Completed,
        unread_message_count: 5,
        ..Session::default()
    };
    let runner = Arc::new(EchoRunner {
        repository: repository.clone(),
        session_id: session.id.clone(),
        attachment_id: String::new(),
        input_was_persisted: AtomicBool::new(false),
    });
    let task = RegisteredTask(Arc::new(RedisStreamTask::new(
        runner,
        app.redis.client.get_multiplexed_async_connection().await?,
    )?));
    session.task_id = Some(task.0.id().to_owned());
    repository.save(session.clone()).await?;
    let mut expected = Vec::new();
    let step = Step {
        id: "step-1".to_owned(),
        description: "执行命令".to_owned(),
        status: ExecutionStatus::Completed,
        ..Step::default()
    };
    for mut event in [
        Event::Title(TitleEvent {
            title: "统一响应".into(),
            ..TitleEvent::default()
        }),
        Event::Plan(PlanEvent {
            plan: Plan {
                steps: vec![step.clone()],
                ..Plan::default()
            },
            ..PlanEvent::default()
        }),
        Event::Step(StepEvent {
            step,
            ..StepEvent::default()
        }),
        Event::Tool(ToolEvent {
            tool_name: "shell".into(),
            function_name: "shell_exec".into(),
            function_args: json!({"session_id": "shell-1"})
                .as_object()
                .unwrap()
                .clone(),
            tool_content: Some(ToolContent::Shell(ShellToolContent {
                console: json!([{"command": "pwd", "output": "/workspace"}]),
            })),
            function_result: Some(ToolResult {
                data: Some(json!({"internal": "保留在领域历史"})),
                ..ToolResult::default()
            }),
            status: ToolEventStatus::Called,
            ..ToolEvent::default()
        }),
        Event::Message(MessageEvent {
            message: "历史输出".into(),
            ..MessageEvent::default()
        }),
        Event::Done(DoneEvent::default()),
    ] {
        let id = task
            .0
            .output_stream()
            .put(serde_json::to_value(&event)?)
            .await?;
        event.set_id(id);
        repository.add_event(&session.id, event.clone()).await?;
        expected.push(serde_json::to_value(AgentSseEvent::from(event))?["data"].clone());
    }
    assert!(task.0.done());

    let response = timeout(Duration::from_secs(5), async {
        app.server
            .post(&format!("/api/sessions/{}/chat", session.id))
            .json(&json!({}))
            .await
    })
    .await?;
    response.assert_status_ok();
    let events = sse_events(&response.text())?;
    assert_eq!(
        events
            .iter()
            .map(|(kind, _)| kind.as_str())
            .collect::<Vec<_>>(),
        ["title", "plan", "step", "tool", "message", "done"]
    );
    // HTTP 的 data 直接承载展示数据；步骤业务 ID 和队列事件 ID 各自保留。
    assert_eq!(events[2].1["id"], "step-1");
    assert!(events[2].1["event_id"].as_str().unwrap().contains('-'));
    assert_eq!(events[2].1["status"], "completed");
    assert_eq!(events[3].1["name"], "shell");
    assert_eq!(events[3].1["function"], "shell_exec");
    assert_eq!(events[3].1["args"], json!({"session_id": "shell-1"}));
    assert_eq!(
        events[3].1["content"],
        json!({"console": [{"command": "pwd", "output": "/workspace"}]})
    );
    assert!(events[3].1.get("function_result").is_none());
    assert_eq!(
        events
            .into_iter()
            .map(|(_, payload)| payload)
            .collect::<Vec<_>>(),
        expected
    );
    let stored = repository.get_by_id(&session.id).await?.unwrap();
    assert_eq!(stored.status, SessionStatus::Completed);
    assert_eq!(stored.unread_message_count, 0);
    Ok(())
}
