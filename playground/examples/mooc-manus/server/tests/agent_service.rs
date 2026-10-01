//! 聊天上半课的编排测试：私有 PostgreSQL + 工厂替身，不执行任务或连接外部服务。

#[path = "support/file_database.rs"]
mod file_database;

use std::{
    net::Ipv4Addr,
    sync::{Arc, Mutex},
};

use anyhow::{ensure, Result};
use async_trait::async_trait;
use futures::StreamExt;
use loco_rs::storage::{drivers, Storage};
use migration::{Migrator, MigratorTrait, SchemaManager};
use serde_json::Value;
use server::{
    application::services::AgentService,
    domain::{
        external::{
            JsonParser, Llm, LlmMessage, Response, ResponseFormat, Sandbox, SandboxFactory,
            SearchEngine, SharedMessageQueue, SharedTask, SharedTaskRunner, Task, TaskFactory,
            Tool, ToolChoice,
        },
        models::{
            A2aConfig, AgentConfig, Event, McpConfig, SearchResults, Session, SessionStatus,
            ToolResult,
        },
        repositories::SessionRepository,
    },
    infrastructure::{
        external::{DockerSandbox, LocoFileStorage},
        repositories::{SeaOrmFileRepository, SeaOrmSessionRepository},
    },
};

struct UnusedDependencies;

#[async_trait]
impl Llm for UnusedDependencies {
    async fn invoke(
        &self,
        _messages: Vec<LlmMessage>,
        _tools: Option<Vec<Tool>>,
        _response_format: Option<ResponseFormat>,
        _tool_choice: Option<ToolChoice>,
    ) -> Result<Response> {
        panic!("本课只创建任务，不调用模型")
    }

    fn model_name(&self) -> String {
        "unused-model".to_owned()
    }
    fn temperature(&self) -> f32 {
        0.7
    }
    fn max_tokens(&self) -> usize {
        8192
    }
}

#[async_trait]
impl JsonParser for UnusedDependencies {
    async fn invoke(&self, _text: &str, _default_value: Option<Value>) -> Result<Value> {
        panic!("本课不解析模型输出")
    }
}

#[async_trait]
impl SearchEngine for UnusedDependencies {
    async fn invoke(
        &self,
        _query: String,
        _date_range: Option<String>,
    ) -> Result<ToolResult<SearchResults>> {
        panic!("本课不执行搜索")
    }
}

struct UnstartedTask(String);

#[async_trait]
impl Task for UnstartedTask {
    async fn invoke(&self) -> Result<()> {
        panic!("本课不得启动 Task")
    }
    fn cancel(&self) -> bool {
        panic!("本课不得取消 Task")
    }
    fn input_stream(&self) -> SharedMessageQueue {
        panic!("消息入队在后续课时实现")
    }
    fn output_stream(&self) -> SharedMessageQueue {
        panic!("输出流读取在后续课时实现")
    }
    fn id(&self) -> &str {
        &self.0
    }
    fn done(&self) -> bool {
        panic!("本课不轮询任务状态")
    }
    fn get(_task_id: &str) -> Result<Option<SharedTask>> {
        panic!("通过工厂查询任务")
    }
    async fn destroy() -> Result<()> {
        panic!("本课不销毁任务注册表")
    }
}

#[derive(Default)]
struct TaskCalls {
    ids: Vec<String>,
    creates: usize,
    found: bool,
    fail_get: bool,
    fail_create: bool,
}

#[derive(Default)]
struct RecordingTaskFactory(Mutex<TaskCalls>);

#[async_trait]
impl TaskFactory for RecordingTaskFactory {
    fn get(&self, task_id: &str) -> Result<Option<SharedTask>> {
        let mut calls = self.0.lock().unwrap();
        calls.ids.push(task_id.to_owned());
        ensure!(!calls.fail_get, "模拟任务查询失败");
        Ok(calls
            .found
            .then(|| Arc::new(UnstartedTask(task_id.to_owned())) as SharedTask))
    }

    async fn create(&self, _runner: SharedTaskRunner) -> Result<SharedTask> {
        let mut calls = self.0.lock().unwrap();
        calls.creates += 1;
        ensure!(!calls.fail_create, "模拟任务创建失败");
        Ok(Arc::new(UnstartedTask(format!(
            "created-task-{}",
            calls.creates
        ))))
    }
}

#[derive(Default)]
struct SandboxCalls {
    ids: Vec<String>,
    creates: usize,
    found: bool,
    fail_get: bool,
    fail_create: bool,
}

struct RecordingSandboxFactory {
    calls: Mutex<SandboxCalls>,
    // 只构造本地对象；get_browser 返回延迟连接的 CDP 包装，不创建 Docker 容器。
    sandbox: DockerSandbox,
}

#[async_trait]
impl SandboxFactory for RecordingSandboxFactory {
    async fn get(&self, id: &str) -> Result<Option<Box<dyn Sandbox>>> {
        let mut calls = self.calls.lock().unwrap();
        calls.ids.push(id.to_owned());
        ensure!(!calls.fail_get, "模拟沙箱查询失败");
        Ok(calls
            .found
            .then(|| Box::new(self.sandbox.clone()) as Box<dyn Sandbox>))
    }

    async fn create(&self) -> Result<Box<dyn Sandbox>> {
        let mut calls = self.calls.lock().unwrap();
        calls.creates += 1;
        ensure!(!calls.fail_create, "模拟沙箱创建失败");
        Ok(Box::new(self.sandbox.clone()))
    }
}

struct Fixture {
    repository: Arc<SeaOrmSessionRepository>,
    file_repository: Arc<SeaOrmFileRepository>,
    tasks: Arc<RecordingTaskFactory>,
    sandboxes: Arc<RecordingSandboxFactory>,
    _database: file_database::TestDatabase,
}

impl Fixture {
    async fn new() -> Result<Self> {
        let database = file_database::TestDatabase::new().await?;
        let manager = SchemaManager::new(&database.db);
        let migrations = Migrator::migrations()
            .into_iter()
            .filter(|migration| {
                matches!(
                    migration.name(),
                    "m20260720_184611_sessions" | "m20260720_191303_fix_sessions_table"
                )
            })
            .collect::<Vec<_>>();
        ensure!(migrations.len() == 2, "没有找到会话表的两条迁移");
        for migration in migrations {
            migration.up(&manager).await?;
        }
        Ok(Self {
            repository: Arc::new(SeaOrmSessionRepository::new(database.db.clone())),
            file_repository: Arc::new(SeaOrmFileRepository::new(database.db.clone())),
            tasks: Arc::default(),
            sandboxes: Arc::new(RecordingSandboxFactory {
                calls: Mutex::default(),
                sandbox: DockerSandbox::new(Ipv4Addr::LOCALHOST, None)?,
            }),
            _database: database,
        })
    }

    fn service(&self) -> AgentService {
        AgentService::new(
            self.repository.clone(),
            Arc::new(UnusedDependencies),
            AgentConfig::default(),
            McpConfig::default(),
            A2aConfig::default(),
            self.sandboxes.clone(),
            self.tasks.clone(),
            Arc::new(UnusedDependencies),
            Arc::new(UnusedDependencies),
            Arc::new(LocoFileStorage::new(
                Arc::new(Storage::single(drivers::mem::new())),
                self.file_repository.clone(),
            )),
            self.file_repository.clone(),
        )
    }

    async fn chat(&self, session: &Session, message: Option<&str>) -> Vec<Event> {
        self.service()
            .chat(
                session.id.clone(),
                message.map(str::to_owned),
                Some(vec!["尚未处理的附件".to_owned()]),
                Some("尚未处理的事件游标".to_owned()),
                Some(123),
            )
            .collect()
            .await
    }
}

fn error_message(events: &[Event]) -> &str {
    match events {
        [Event::Error(error)] => &error.error,
        other => panic!("应仅返回一个错误事件，实际为 {other:?}"),
    }
}

#[tokio::test]
async fn missing_or_empty_messages_only_look_up_the_existing_task() -> Result<()> {
    let fixture = Fixture::new().await?;
    fixture.tasks.0.lock().unwrap().found = true;
    let session = Session {
        task_id: Some("existing-task".to_owned()),
        ..Session::default()
    };
    fixture.repository.save(session.clone()).await?;
    let before = fixture.repository.get_by_id(&session.id).await?.unwrap();
    for message in [None, Some("")] {
        assert!(fixture.chat(&session, message).await.is_empty());
        assert_eq!(
            fixture.repository.get_by_id(&session.id).await?.unwrap(),
            before
        );
    }
    assert_eq!(
        fixture.tasks.0.lock().unwrap().ids,
        ["existing-task", "existing-task"]
    );
    assert_eq!(fixture.tasks.0.lock().unwrap().creates, 0);
    let sandboxes = fixture.sandboxes.calls.lock().unwrap();
    assert!(sandboxes.ids.is_empty());
    assert_eq!(sandboxes.creates, 0);
    Ok(())
}

#[tokio::test]
async fn running_sessions_never_create_tasks_even_when_the_task_is_missing() -> Result<()> {
    let fixture = Fixture::new().await?;
    for (task_id, found) in [
        (Some("existing-task"), true),
        (Some("missing-task"), false),
        (None, false),
        (Some(""), false),
    ] {
        fixture.tasks.0.lock().unwrap().found = found;
        let session = Session {
            task_id: task_id.map(str::to_owned),
            status: SessionStatus::Running,
            ..Session::default()
        };
        fixture.repository.save(session.clone()).await?;
        let before = fixture.repository.get_by_id(&session.id).await?.unwrap();
        assert!(fixture.chat(&session, Some("继续任务")).await.is_empty());
        assert_eq!(
            fixture.repository.get_by_id(&session.id).await?.unwrap(),
            before
        );
    }
    assert_eq!(
        fixture.tasks.0.lock().unwrap().ids,
        ["existing-task", "missing-task"]
    );
    assert_eq!(fixture.tasks.0.lock().unwrap().creates, 0);
    let sandboxes = fixture.sandboxes.calls.lock().unwrap();
    assert!(sandboxes.ids.is_empty());
    assert_eq!(sandboxes.creates, 0);
    Ok(())
}

#[tokio::test]
async fn non_running_sessions_create_unstarted_tasks_and_reuse_the_sandbox() -> Result<()> {
    let fixture = Fixture::new().await?;
    fixture.tasks.0.lock().unwrap().found = true;
    fixture.sandboxes.calls.lock().unwrap().found = true;
    let sandbox_id = fixture.sandboxes.sandbox.id();
    for (index, status) in [
        SessionStatus::Pending,
        SessionStatus::Waiting,
        SessionStatus::Completed,
    ]
    .into_iter()
    .enumerate()
    {
        let session = Session {
            task_id: Some("previous-task".to_owned()),
            sandbox_id: Some(sandbox_id.to_owned()),
            title: "保留会话内容".to_owned(),
            unread_message_count: 3,
            status,
            ..Session::default()
        };
        fixture.repository.save(session.clone()).await?;
        // 空白字符串沿用课程真值语义；创建 Stream 时还不能创建任务。
        let stream =
            fixture
                .service()
                .chat(session.id.clone(), Some("   ".to_owned()), None, None, None);
        assert_eq!(fixture.tasks.0.lock().unwrap().creates, index);
        assert!(stream.collect::<Vec<_>>().await.is_empty());
        let saved = fixture.repository.get_by_id(&session.id).await?.unwrap();
        assert_eq!(saved.task_id, Some(format!("created-task-{}", index + 1)));
        assert_eq!(saved.sandbox_id, session.sandbox_id);
        assert_eq!(saved.status, status);
        assert_eq!(saved.title, session.title);
        assert_eq!(saved.unread_message_count, 3);
        assert!(saved.events.is_empty() && saved.files.is_empty() && saved.memories.is_empty());
    }
    assert_eq!(
        fixture.tasks.0.lock().unwrap().ids,
        ["previous-task", "previous-task", "previous-task"]
    );
    assert_eq!(fixture.tasks.0.lock().unwrap().creates, 3);
    let sandboxes = fixture.sandboxes.calls.lock().unwrap();
    assert_eq!(sandboxes.ids, [sandbox_id, sandbox_id, sandbox_id]);
    assert_eq!(sandboxes.creates, 0);
    Ok(())
}

#[tokio::test]
async fn absent_or_released_sandboxes_are_rebuilt_and_their_ids_are_saved() -> Result<()> {
    let fixture = Fixture::new().await?;
    for (index, sandbox_id) in [None, Some("released-sandbox"), Some("")]
        .into_iter()
        .enumerate()
    {
        let session = Session {
            sandbox_id: sandbox_id.map(str::to_owned),
            ..Session::default()
        };
        fixture.repository.save(session.clone()).await?;
        assert!(fixture.chat(&session, Some("新任务")).await.is_empty());
        let saved = fixture.repository.get_by_id(&session.id).await?.unwrap();
        assert_eq!(
            saved.sandbox_id.as_deref(),
            Some(fixture.sandboxes.sandbox.id())
        );
        assert_eq!(saved.task_id, Some(format!("created-task-{}", index + 1)));
        assert_eq!(saved.status, SessionStatus::Pending);
    }
    assert!(fixture.tasks.0.lock().unwrap().ids.is_empty());
    assert_eq!(fixture.tasks.0.lock().unwrap().creates, 3);
    let sandboxes = fixture.sandboxes.calls.lock().unwrap();
    assert_eq!(sandboxes.ids, ["released-sandbox"]);
    assert_eq!(sandboxes.creates, 3);
    Ok(())
}

#[tokio::test]
async fn lookup_and_sandbox_failures_are_persisted_as_the_returned_error_event() -> Result<()> {
    let fixture = Fixture::new().await?;
    for failure in ["task-get", "sandbox-get", "sandbox-create"] {
        fixture.tasks.0.lock().unwrap().fail_get = failure == "task-get";
        {
            let mut calls = fixture.sandboxes.calls.lock().unwrap();
            calls.fail_get = failure == "sandbox-get";
            calls.fail_create = failure == "sandbox-create";
        }
        let session = Session {
            task_id: (failure == "task-get").then(|| "unavailable-task".to_owned()),
            sandbox_id: (failure == "sandbox-get").then(|| "unavailable-sandbox".to_owned()),
            ..Session::default()
        };
        fixture.repository.save(session.clone()).await?;
        let events = fixture.chat(&session, Some("新任务")).await;
        let expected = match failure {
            "task-get" => "模拟任务查询失败",
            "sandbox-get" => "模拟沙箱查询失败",
            _ => "模拟沙箱创建失败",
        };
        assert_eq!(error_message(&events), expected);
        let saved = fixture.repository.get_by_id(&session.id).await?.unwrap();
        assert_eq!(saved.events, events);
        assert_eq!(saved.task_id, session.task_id);
        assert_eq!(saved.status, SessionStatus::Pending);
    }
    assert_eq!(fixture.tasks.0.lock().unwrap().creates, 0);
    assert_eq!(fixture.sandboxes.calls.lock().unwrap().creates, 1);
    Ok(())
}

#[tokio::test]
async fn task_creation_failure_keeps_the_saved_sandbox_and_original_task_id() -> Result<()> {
    let fixture = Fixture::new().await?;
    fixture.tasks.0.lock().unwrap().fail_create = true;
    let session = Session {
        task_id: Some("previous-task".to_owned()),
        ..Session::default()
    };
    fixture.repository.save(session.clone()).await?;
    let events = fixture.chat(&session, Some("新任务")).await;
    assert_eq!(
        error_message(&events),
        format!("会话[{}]创建任务失败", session.id)
    );
    let saved = fixture.repository.get_by_id(&session.id).await?.unwrap();
    // 新沙箱必须先保存；后续创建 Task 失败也不能丢失它的标识。
    assert_eq!(
        saved.sandbox_id.as_deref(),
        Some(fixture.sandboxes.sandbox.id())
    );
    assert_eq!(saved.task_id, session.task_id);
    assert_eq!(saved.events, events);
    assert_eq!(saved.status, SessionStatus::Pending);
    assert_eq!(fixture.tasks.0.lock().unwrap().creates, 1);
    assert_eq!(fixture.sandboxes.calls.lock().unwrap().creates, 1);
    Ok(())
}

#[tokio::test]
async fn missing_session_returns_the_original_error_when_event_persistence_also_fails() -> Result<()>
{
    let fixture = Fixture::new().await?;
    let session = Session::default();
    let events = fixture.chat(&session, Some("新任务")).await;
    assert_eq!(error_message(&events), "任务会话不存在, 请核实后重试");
    assert!(fixture.repository.get_all().await?.is_empty());
    assert!(fixture.tasks.0.lock().unwrap().ids.is_empty());
    assert_eq!(fixture.tasks.0.lock().unwrap().creates, 0);
    let sandboxes = fixture.sandboxes.calls.lock().unwrap();
    assert!(sandboxes.ids.is_empty());
    assert_eq!(sandboxes.creates, 0);
    Ok(())
}
