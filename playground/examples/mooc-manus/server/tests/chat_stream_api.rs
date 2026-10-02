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
use tokio::{sync::Notify, time::timeout};

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
        Self::with_http_transport(false).await
    }

    async fn with_http_transport(http_transport: bool) -> Result<Self> {
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
        let server = if http_transport {
            // 随机监听端口，允许客户端在响应完成前增量读取 SSE。
            TestServer::builder().http_transport().build(router)?
        } else {
            TestServer::new(router)?
        };
        Ok(Self {
            server,
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

/// 暂停在工具执行阶段，验证 Calling 已经通过 HTTP 传播并写入数据库。
struct GatedToolRunner {
    repository: Arc<SeaOrmSessionRepository>,
    session_id: String,
    entered: Arc<Notify>,
    release: Arc<Notify>,
    resumed: AtomicBool,
}

impl GatedToolRunner {
    async fn publish(&self, task: &SharedTask, mut event: Event) -> Result<()> {
        let id = task
            .output_stream()
            .put(serde_json::to_value(&event)?)
            .await?;
        event.set_id(id);
        self.repository.add_event(&self.session_id, event).await
    }
}

#[async_trait]
impl TaskRunner for GatedToolRunner {
    async fn invoke(&self, task: SharedTask) -> Result<()> {
        let (_, payload) = task.input_stream().pop().await?.context("输入流缺少消息")?;
        let Event::Message(message) = serde_json::from_value::<Event>(payload)? else {
            anyhow::bail!("输入流应包含消息事件");
        };
        ensure!(message.role == MessageRole::User && message.message == MESSAGE);

        let mut tool = ToolEvent {
            tool_name: "shell".into(),
            function_name: "shell_exec".into(),
            function_args: json!({"session_id": "shell-1"})
                .as_object()
                .unwrap()
                .clone(),
            status: ToolEventStatus::Calling,
            ..ToolEvent::default()
        };
        self.publish(&task, Event::Tool(tool.clone())).await?;
        // 通知只发生在 Calling 已保存之后；释放门闩前任务保持运行。
        self.entered.notify_one();
        self.release.notified().await;
        self.resumed.store(true, Ordering::SeqCst);

        tool.status = ToolEventStatus::Called;
        tool.tool_content = Some(ToolContent::Shell(ShellToolContent {
            console: json!([{"command": "pwd", "output": "/workspace"}]),
        }));
        self.publish(&task, Event::Tool(tool)).await?;
        self.publish(&task, Event::Done(DoneEvent::default()))
            .await?;
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

/// 提前返回或断言失败时也释放门闩，任务注册守卫随后取消自身任务。
struct ReleaseGate(Arc<Notify>);

impl Drop for ReleaseGate {
    fn drop(&mut self) {
        self.0.notify_one();
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

async fn next_sse_frame(
    response: &mut reqwest::Response,
    buffered: &mut Vec<u8>,
) -> Result<String> {
    loop {
        if let Some(end) = buffered.windows(2).position(|bytes| bytes == b"\n\n") {
            // TCP 分块边界可以落在 UTF-8 字符或 SSE 帧中间，完整帧后再解码。
            return Ok(String::from_utf8(buffered.drain(..end + 2).collect())?);
        }
        let chunk = response.chunk().await?.context("SSE 在完整帧前结束")?;
        buffered.extend_from_slice(&chunk);
    }
}

#[tokio::test]
#[serial]
async fn chat_stream_delivers_calling_over_http_while_tool_is_blocked() -> Result<()> {
    let app = TestApp::with_http_transport(true).await?;
    let repository = Arc::new(app.repository());
    let mut session = Session {
        status: SessionStatus::Running,
        ..Session::default()
    };
    let runner = Arc::new(GatedToolRunner {
        repository: repository.clone(),
        session_id: session.id.clone(),
        entered: Arc::new(Notify::new()),
        release: Arc::new(Notify::new()),
        resumed: AtomicBool::new(false),
    });
    let task = RegisteredTask(Arc::new(RedisStreamTask::new(
        runner.clone(),
        app.redis.client.get_multiplexed_async_connection().await?,
    )?));
    let release_gate = ReleaseGate(runner.release.clone());
    session.task_id = Some(task.0.id().to_owned());
    repository.save(session.clone()).await?;

    let url = app
        .server
        .server_url(&format!("/api/sessions/{}/chat", session.id))?;
    let mut response = timeout(Duration::from_secs(5), async {
        reqwest::Client::new()
            .post(url)
            .json(&json!({"message": MESSAGE}))
            .send()
            .await
    })
    .await
    .context("聊天 SSE 应及时返回响应头")??;
    ensure!(response.status().is_success());
    ensure!(response.headers()["content-type"] == "text/event-stream");
    timeout(Duration::from_secs(5), runner.entered.notified())
        .await
        .context("任务应在 Calling 持久化后进入工具门闩")?;

    let mut buffered = Vec::new();
    let first_frame = timeout(
        Duration::from_secs(5),
        next_sse_frame(&mut response, &mut buffered),
    )
    .await
    .context("工具尚未完成时，客户端应收到 Calling 帧")??;
    let first = sse_events(&first_frame)?;
    // 在门闩关闭时记录实际状态，随后先释放任务，再验收这些快照。
    let was_done = task.0.done();
    let resumed = runner.resumed.load(Ordering::SeqCst);
    let during = repository.get_by_id(&session.id).await?.unwrap();
    drop(release_gate);

    timeout(Duration::from_secs(5), async {
        while let Some(chunk) = response.chunk().await? {
            buffered.extend_from_slice(&chunk);
        }
        while !task.0.done() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await
    .context("工具释放后应收到 Called、Done 并完成任务")??;

    assert!(!was_done && !resumed);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].0, "tool");
    assert_eq!(first[0].1["status"], "calling");
    assert_eq!(during.status, SessionStatus::Running);
    assert_eq!(during.events.len(), 2);
    let saved_calling = serde_json::to_value(AgentSseEvent::from(during.events[1].clone()))?;
    assert_eq!(first[0].1, saved_calling["data"]);

    let mut received = first;
    received.extend(sse_events(std::str::from_utf8(&buffered)?)?);
    assert_eq!(
        received
            .iter()
            .map(|(kind, data)| (kind.as_str(), data["status"].as_str()))
            .collect::<Vec<_>>(),
        [
            ("tool", Some("calling")),
            ("tool", Some("called")),
            ("done", None)
        ]
    );
    assert_eq!(
        received[1].1["content"],
        json!({"console": [{"command": "pwd", "output": "/workspace"}]})
    );
    let stored = repository.get_by_id(&session.id).await?.unwrap();
    assert_eq!(stored.events.len(), 4);
    for ((_, data), event) in received.iter().zip(&stored.events[1..]) {
        let saved = serde_json::to_value(AgentSseEvent::from(event.clone()))?;
        assert_eq!(data, &saved["data"]);
    }
    assert_eq!(stored.status, SessionStatus::Completed);
    assert_eq!(stored.unread_message_count, 0);
    Ok(())
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
