//! 聊天服务编排与订阅测试：私有 PostgreSQL + 内存任务/队列，不连接模型或外部服务。

#[path = "support/file_database.rs"]
mod file_database;

use std::{
    collections::HashMap,
    net::Ipv4Addr,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use anyhow::{ensure, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use loco_rs::storage::{drivers, Storage};
use migration::{Migrator, MigratorTrait, SchemaManager};
use sea_orm::ConnectionTrait;
use serde_json::{json, Value};
use server::{
    application::services::AgentService,
    domain::{
        external::{
            JsonParser, Llm, LlmMessage, MessageQueue, Response, ResponseFormat, Sandbox,
            SandboxFactory, SearchEngine, SharedMessageQueue, SharedTask, SharedTaskRunner, Task,
            TaskFactory, Tool, ToolChoice,
        },
        models::{
            A2aConfig, AgentConfig, DoneEvent, ErrorEvent, Event, File, McpConfig, Memory,
            MessageEvent, MessageRole, SearchResults, Session, SessionStatus, ToolResult,
            WaitEvent,
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
        panic!("服务测试不调用真实模型")
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

#[derive(Default)]
struct QueueState {
    entries: Vec<(String, Value)>,
    reads: Vec<(Option<String>, Option<usize>)>,
    fail_put: bool,
    fail_get: bool,
    // 模拟空读返回时任务刚结束；下一次读取才能看到已写入的终止事件。
    finish_on_empty: Option<Arc<AtomicBool>>,
}

#[derive(Default)]
struct RecordingQueue(Mutex<QueueState>);

fn queue_id(id: &str) -> (u64, u64) {
    let (timestamp, sequence) = id.split_once('-').unwrap();
    (timestamp.parse().unwrap(), sequence.parse().unwrap())
}

#[async_trait]
impl MessageQueue for RecordingQueue {
    async fn put(&self, message: Value) -> Result<String> {
        let mut state = self.0.lock().unwrap();
        ensure!(!state.fail_put, "模拟队列写入失败");
        let id = format!("{}-0", state.entries.len() + 1);
        state.entries.push((id.clone(), message));
        Ok(id)
    }

    async fn get(
        &self,
        start_id: Option<&str>,
        block_ms: Option<usize>,
    ) -> Result<Option<(String, Value)>> {
        let mut state = self.0.lock().unwrap();
        state.reads.push((start_id.map(str::to_owned), block_ms));
        assert_eq!(block_ms, None, "共享连接必须使用非阻塞读取");
        ensure!(!state.fail_get, "模拟队列读取失败");
        if let Some(done) = state.finish_on_empty.take() {
            done.store(true, Ordering::SeqCst);
            return Ok(None);
        }
        let cursor = start_id.map(queue_id).unwrap_or_default();
        Ok(state
            .entries
            .iter()
            .find(|(id, _)| queue_id(id) > cursor)
            .cloned())
    }

    async fn pop(&self) -> Result<Option<(String, Value)>> {
        panic!("服务层不消费任务输入队列")
    }
    async fn clear(&self) -> Result<()> {
        panic!("订阅不能清空后台任务队列")
    }
    async fn is_empty(&self) -> Result<bool> {
        Ok(self.0.lock().unwrap().entries.is_empty())
    }
    async fn size(&self) -> Result<usize> {
        Ok(self.0.lock().unwrap().entries.len())
    }
    async fn delete_message(&self, _message_id: &str) -> Result<bool> {
        panic!("订阅不删除队列事件")
    }
}

struct RecordingTask {
    id: String,
    input: Arc<RecordingQueue>,
    output: Arc<RecordingQueue>,
    done: Arc<AtomicBool>,
    invokes: AtomicUsize,
    repository: Arc<SeaOrmSessionRepository>,
    persisted_at_invoke: Mutex<Vec<Event>>,
}

impl RecordingTask {
    fn new(id: String, repository: Arc<SeaOrmSessionRepository>) -> Self {
        Self {
            id,
            repository,
            input: Arc::default(),
            output: Arc::default(),
            done: Arc::new(AtomicBool::new(true)),
            invokes: AtomicUsize::new(0),
            persisted_at_invoke: Mutex::default(),
        }
    }
}

#[async_trait]
impl Task for RecordingTask {
    async fn invoke(&self) -> Result<()> {
        self.invokes.fetch_add(1, Ordering::SeqCst);
        // 在 invoke 边界观察真实数据库，验证人类事件已先持久化。
        let events = self
            .repository
            .get_all()
            .await?
            .into_iter()
            .filter(|session| session.task_id.as_deref() == Some(self.id.as_str()))
            .flat_map(|session| session.events)
            .collect();
        *self.persisted_at_invoke.lock().unwrap() = events;
        Ok(())
    }
    fn cancel(&self) -> bool {
        panic!("结束 HTTP 订阅不能取消后台 Task")
    }
    fn input_stream(&self) -> SharedMessageQueue {
        self.input.clone()
    }
    fn output_stream(&self) -> SharedMessageQueue {
        self.output.clone()
    }
    fn id(&self) -> &str {
        &self.id
    }
    fn done(&self) -> bool {
        self.done.load(Ordering::SeqCst)
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
    registry: HashMap<String, Arc<RecordingTask>>,
}

struct RecordingTaskFactory(Mutex<TaskCalls>, Arc<SeaOrmSessionRepository>);

impl RecordingTaskFactory {
    fn task(&self, id: &str) -> Arc<RecordingTask> {
        self.0
            .lock()
            .unwrap()
            .registry
            .entry(id.to_owned())
            .or_insert_with(|| Arc::new(RecordingTask::new(id.to_owned(), self.1.clone())))
            .clone()
    }
}

#[async_trait]
impl TaskFactory for RecordingTaskFactory {
    fn get(&self, task_id: &str) -> Result<Option<SharedTask>> {
        let mut calls = self.0.lock().unwrap();
        calls.ids.push(task_id.to_owned());
        ensure!(!calls.fail_get, "模拟任务查询失败");
        if let Some(task) = calls.registry.get(task_id) {
            return Ok(Some(task.clone()));
        }
        if !calls.found {
            return Ok(None);
        }
        let task = Arc::new(RecordingTask::new(task_id.to_owned(), self.1.clone()));
        calls.registry.insert(task_id.to_owned(), task.clone());
        Ok(Some(task))
    }

    async fn create(&self, _runner: SharedTaskRunner) -> Result<SharedTask> {
        let mut calls = self.0.lock().unwrap();
        calls.creates += 1;
        ensure!(!calls.fail_create, "模拟任务创建失败");
        let id = format!("created-task-{}", calls.creates);
        let task = Arc::new(RecordingTask::new(id.clone(), self.1.clone()));
        calls.registry.insert(id, task.clone());
        Ok(task)
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
    database: file_database::TestDatabase,
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
        let repository = Arc::new(SeaOrmSessionRepository::new(database.db.clone()));
        Ok(Self {
            repository: repository.clone(),
            file_repository: Arc::new(SeaOrmFileRepository::new(database.db.clone())),
            tasks: Arc::new(RecordingTaskFactory(Mutex::default(), repository)),
            sandboxes: Arc::new(RecordingSandboxFactory {
                calls: Mutex::default(),
                sandbox: DockerSandbox::new(Ipv4Addr::LOCALHOST, None)?,
            }),
            database,
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
        tokio::time::timeout(
            Duration::from_secs(2),
            self.service()
                .chat(
                    session.id.clone(),
                    message.map(str::to_owned),
                    None,
                    None,
                    None,
                )
                .collect(),
        )
        .await
        .expect("测试聊天流应在期限内结束")
    }

    async fn subscription(&self) -> Result<(Session, Arc<RecordingTask>)> {
        let task_id = uuid::Uuid::new_v4().to_string();
        let session = Session {
            task_id: Some(task_id.clone()),
            status: SessionStatus::Running,
            unread_message_count: 5,
            ..Session::default()
        };
        self.repository.save(session.clone()).await?;
        let task = self.tasks.task(&task_id);
        Ok((session, task))
    }
}

fn error_message(events: &[Event]) -> &str {
    match events {
        [Event::Error(error)] => &error.error,
        other => panic!("应仅返回一个错误事件，实际为 {other:?}"),
    }
}

fn reply(message: &str) -> Event {
    Event::Message(MessageEvent {
        message: message.to_owned(),
        ..MessageEvent::default()
    })
}

async fn queue_event(queue: &RecordingQueue, mut event: Event) -> Result<Event> {
    let id = queue.put(serde_json::to_value(&event)?).await?;
    event.set_id(id);
    Ok(event)
}

#[tokio::test]
async fn missing_or_empty_messages_only_look_up_the_existing_task() -> Result<()> {
    let fixture = Fixture::new().await?;
    for (status, task_id, found) in [
        (SessionStatus::Pending, Some("existing-task"), true),
        (SessionStatus::Running, Some("running-task"), true),
        (SessionStatus::Running, Some("missing-task"), false),
        (SessionStatus::Running, None, false),
        (SessionStatus::Running, Some(""), false),
    ] {
        fixture.tasks.0.lock().unwrap().found = found;
        let session = Session {
            task_id: task_id.map(str::to_owned),
            unread_message_count: 3,
            status,
            ..Session::default()
        };
        fixture.repository.save(session.clone()).await?;
        let before = fixture.repository.get_by_id(&session.id).await?.unwrap();
        for message in [None, Some("")] {
            // 仅订阅或空消息时，Running 会话缺失 Task 也只执行查询。
            assert!(fixture.chat(&session, message).await.is_empty());
            let saved = fixture.repository.get_by_id(&session.id).await?.unwrap();
            let mut expected = before.clone();
            expected.unread_message_count = 0;
            expected.updated_at = saved.updated_at;
            assert_eq!(saved, expected);
        }
    }
    for task_id in ["existing-task", "running-task"] {
        let task = fixture.tasks.task(task_id);
        assert_eq!(task.invokes.load(Ordering::SeqCst), 0);
        assert!(task.input.0.lock().unwrap().entries.is_empty());
    }
    assert_eq!(
        fixture.tasks.0.lock().unwrap().ids,
        [
            "existing-task",
            "existing-task",
            "running-task",
            "running-task",
            "missing-task",
            "missing-task"
        ]
    );
    assert_eq!(fixture.tasks.0.lock().unwrap().creates, 0);
    let sandboxes = fixture.sandboxes.calls.lock().unwrap();
    assert!(sandboxes.ids.is_empty());
    assert_eq!(sandboxes.creates, 0);
    Ok(())
}

#[tokio::test]
async fn running_sessions_reuse_existing_tasks() -> Result<()> {
    let fixture = Fixture::new().await?;
    fixture.tasks.0.lock().unwrap().found = true;
    let session = Session {
        task_id: Some("existing-task".to_owned()),
        status: SessionStatus::Running,
        ..Session::default()
    };
    fixture.repository.save(session.clone()).await?;
    assert!(fixture.chat(&session, Some("继续任务")).await.is_empty());
    let saved = fixture.repository.get_by_id(&session.id).await?.unwrap();
    assert_eq!(saved.task_id, session.task_id);
    assert_eq!(saved.status, SessionStatus::Running);
    assert_eq!(saved.latest_message, "继续任务");
    assert!(matches!(saved.events.as_slice(), [Event::Message(event)]
        if event.role == MessageRole::User && event.message == "继续任务"));
    let task = fixture.tasks.task("existing-task");
    assert_eq!(task.invokes.load(Ordering::SeqCst), 1);
    assert_eq!(task.input.0.lock().unwrap().entries.len(), 1);
    assert_eq!(*task.persisted_at_invoke.lock().unwrap(), saved.events);
    assert_eq!(fixture.tasks.0.lock().unwrap().ids, ["existing-task"]);
    assert_eq!(fixture.tasks.0.lock().unwrap().creates, 0);
    let sandboxes = fixture.sandboxes.calls.lock().unwrap();
    assert!(sandboxes.ids.is_empty());
    assert_eq!(sandboxes.creates, 0);
    Ok(())
}

#[tokio::test]
async fn running_sessions_rebuild_missing_tasks_and_preserve_existing_session_data() -> Result<()> {
    let fixture = Fixture::new().await?;
    fixture.sandboxes.calls.lock().unwrap().found = true;
    let sandbox_id = fixture.sandboxes.sandbox.id();
    let timestamp = DateTime::<Utc>::from_timestamp(1_800_000_000, 0).unwrap();
    for (index, task_id) in [Some("missing-task"), None, Some("")]
        .into_iter()
        .enumerate()
    {
        let session = Session {
            task_id: task_id.map(str::to_owned),
            sandbox_id: Some(sandbox_id.to_owned()),
            title: "保留会话内容".to_owned(),
            unread_message_count: 3,
            latest_message: "上一轮消息".to_owned(),
            events: vec![reply("保留历史事件")],
            files: vec![File {
                id: "existing-file".to_owned(),
                filename: "历史附件.txt".to_owned(),
                ..File::default()
            }],
            memories: HashMap::from([(
                "agent".to_owned(),
                Memory {
                    messages: vec![serde_json::from_value(json!({
                        "role": "user", "content": "保留历史记忆"
                    }))?],
                },
            )]),
            status: SessionStatus::Running,
            ..Session::default()
        };
        fixture.repository.save(session.clone()).await?;
        let before = fixture.repository.get_by_id(&session.id).await?.unwrap();
        // 进程重启后注册表中的 Task 可丢失；新消息重建任务并沿用已有沙箱。
        let stream = fixture.service().chat(
            session.id.clone(),
            Some("继续任务".to_owned()),
            None,
            None,
            Some(timestamp),
        );
        assert!(stream.collect::<Vec<_>>().await.is_empty());
        let saved = fixture.repository.get_by_id(&session.id).await?.unwrap();
        let [_, Event::Message(message)] = saved.events.as_slice() else {
            panic!("应保留历史事件并追加人类消息事件")
        };
        assert_eq!(message.role, MessageRole::User);
        assert_eq!(message.message, "继续任务");
        assert_eq!(message.base.id, "1-0");
        assert!(message.attachments.is_empty());
        let mut expected = before;
        expected.task_id = Some(format!("created-task-{}", index + 1));
        expected.latest_message = "继续任务".to_owned();
        expected.latest_message_at = Some(timestamp);
        expected.unread_message_count = 0;
        expected.events.push(Event::Message(message.clone()));
        expected.updated_at = saved.updated_at;
        assert_eq!(saved, expected);
        let task = fixture.tasks.task(saved.task_id.as_deref().unwrap());
        assert_eq!(task.invokes.load(Ordering::SeqCst), 1);
        assert_eq!(*task.persisted_at_invoke.lock().unwrap(), saved.events);
        let entries = task.input.0.lock().unwrap().entries.clone();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].1["role"], "user");
        assert_eq!(entries[0].1["message"], "继续任务");
        assert_eq!(fixture.tasks.0.lock().unwrap().creates, index + 1);
    }
    assert_eq!(fixture.tasks.0.lock().unwrap().ids, ["missing-task"]);
    assert_eq!(fixture.tasks.0.lock().unwrap().creates, 3);
    let sandboxes = fixture.sandboxes.calls.lock().unwrap();
    assert_eq!(sandboxes.ids, [sandbox_id, sandbox_id, sandbox_id]);
    assert_eq!(sandboxes.creates, 0);
    Ok(())
}

#[tokio::test]
async fn non_running_sessions_create_and_invoke_tasks_while_reusing_the_sandbox() -> Result<()> {
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
        assert_eq!(saved.unread_message_count, 0);
        assert_eq!(saved.latest_message, "   ");
        assert!(
            matches!(saved.events.as_slice(), [Event::Message(event)] if event.message == "   ")
        );
        assert!(saved.files.is_empty() && saved.memories.is_empty());
        let task = fixture.tasks.task(saved.task_id.as_deref().unwrap());
        assert_eq!(task.invokes.load(Ordering::SeqCst), 1);
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
        assert_eq!(saved.latest_message, "新任务");
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

#[tokio::test]
async fn enqueues_user_attachments_and_persists_the_queue_id_before_invoking() -> Result<()> {
    let fixture = Fixture::new().await?;
    let (session, task) = fixture.subscription().await?;
    let timestamp = DateTime::<Utc>::from_timestamp(1_800_000_000, 0).unwrap();
    let stream = fixture.service().chat(
        session.id.clone(),
        Some("读取附件".to_owned()),
        Some(vec!["file-1".to_owned(), "file-2".to_owned()]),
        None,
        Some(timestamp),
    );
    assert!(task.input.0.lock().unwrap().entries.is_empty());
    assert_eq!(task.invokes.load(Ordering::SeqCst), 0);
    assert!(stream.collect::<Vec<_>>().await.is_empty());

    let saved = fixture.repository.get_by_id(&session.id).await?.unwrap();
    assert_eq!(saved.latest_message, "读取附件");
    assert_eq!(saved.latest_message_at, Some(timestamp));
    assert_eq!(saved.unread_message_count, 0);
    assert_eq!(saved.status, SessionStatus::Running);
    let [Event::Message(message)] = saved.events.as_slice() else {
        panic!("应保存人类消息事件")
    };
    assert_eq!(message.role, MessageRole::User);
    assert_eq!(message.message, "读取附件");
    assert_eq!(message.base.id, "1-0");
    assert_eq!(
        message
            .attachments
            .iter()
            .map(|file| file.id.as_str())
            .collect::<Vec<_>>(),
        ["file-1", "file-2"]
    );
    assert!(message
        .attachments
        .iter()
        .all(|file| file.filename.is_empty() && file.filepath.is_empty() && file.size == 0));
    let entries = task.input.0.lock().unwrap().entries.clone();
    assert_eq!(entries.len(), 1);
    assert!(
        entries[0].1.is_object(),
        "队列协议接收 JSON 对象，不能双重编码成字符串"
    );
    assert_eq!(entries[0].1["role"], "user");
    assert_eq!(entries[0].1["message"], "读取附件");
    assert_eq!(task.invokes.load(Ordering::SeqCst), 1);
    assert_eq!(*task.persisted_at_invoke.lock().unwrap(), saved.events);
    assert_eq!(fixture.tasks.0.lock().unwrap().creates, 0);
    Ok(())
}

#[tokio::test]
async fn completed_tasks_drain_outputs_through_each_terminal_event_without_repersisting_them(
) -> Result<()> {
    let fixture = Fixture::new().await?;
    for terminal in [
        Event::Done(DoneEvent::default()),
        Event::Error(ErrorEvent {
            error: "后台任务错误".to_owned(),
            ..ErrorEvent::default()
        }),
        Event::Wait(WaitEvent::default()),
    ] {
        let (session, task) = fixture.subscription().await?;
        let first = queue_event(&task.output, reply("任务回复")).await?;
        let terminal = queue_event(&task.output, terminal).await?;
        queue_event(&task.output, reply("下一轮事件")).await?;
        assert!(task.done());
        let events = fixture.chat(&session, None).await;
        assert_eq!(events, vec![first, terminal]);
        assert_eq!(
            task.output.0.lock().unwrap().reads,
            vec![(None, None), (Some("1-0".to_owned()), None)]
        );
        assert_eq!(task.invokes.load(Ordering::SeqCst), 0);
        assert!(task.input.0.lock().unwrap().entries.is_empty());
        let saved = fixture.repository.get_by_id(&session.id).await?.unwrap();
        assert_eq!(saved.unread_message_count, 0);
        assert!(
            saved.events.is_empty(),
            "输出历史由 Runner 保存，订阅不能重复追加"
        );
    }
    Ok(())
}

#[tokio::test]
async fn an_empty_read_racing_with_task_completion_still_delivers_the_final_event() -> Result<()> {
    let fixture = Fixture::new().await?;
    let (session, task) = fixture.subscription().await?;
    task.done.store(false, Ordering::SeqCst);
    let done = queue_event(&task.output, Event::Done(DoneEvent::default())).await?;
    task.output.0.lock().unwrap().finish_on_empty = Some(task.done.clone());
    let events = tokio::time::timeout(Duration::from_secs(2), fixture.chat(&session, None)).await?;
    assert_eq!(events, vec![done]);
    assert!(task.done());
    assert_eq!(
        task.output.0.lock().unwrap().reads,
        vec![(None, None), (None, None)]
    );
    Ok(())
}

#[tokio::test]
async fn subscription_cursor_reads_only_newer_events_and_replaces_payload_ids() -> Result<()> {
    let fixture = Fixture::new().await?;
    let (session, task) = fixture.subscription().await?;
    queue_event(&task.output, reply("已经读过")).await?;
    let next = queue_event(&task.output, reply("尚未读过")).await?;
    let done = queue_event(&task.output, Event::Done(DoneEvent::default())).await?;
    let events = fixture
        .service()
        .chat(session.id, None, None, Some("1-0".to_owned()), None)
        .collect::<Vec<_>>()
        .await;
    assert_eq!(events, vec![next, done]);
    assert_eq!(
        task.output.0.lock().unwrap().reads,
        vec![
            (Some("1-0".to_owned()), None),
            (Some("2-0".to_owned()), None)
        ]
    );
    assert_eq!(task.invokes.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn dropping_a_waiting_subscription_keeps_the_task_and_new_unread_messages() -> Result<()> {
    let fixture = Fixture::new().await?;
    let (session, task) = fixture.subscription().await?;
    task.done.store(false, Ordering::SeqCst);
    let first = queue_event(&task.output, reply("已传递的消息")).await?;
    let mut stream = fixture
        .service()
        .chat(session.id.clone(), None, None, None, None);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), stream.next()).await?,
        Some(first)
    );
    assert_eq!(
        fixture
            .repository
            .get_by_id(&session.id)
            .await?
            .unwrap()
            .unread_message_count,
        0
    );
    // 在空读等待中断开，覆盖丢弃正在等待的 Stream，而不仅是正常结束。
    assert!(
        tokio::time::timeout(Duration::from_millis(20), stream.next())
            .await
            .is_err()
    );
    let reads_before_drop = task.output.0.lock().unwrap().reads.len();
    drop(stream);

    // 注册表仍持有任务；模拟后台在订阅结束后继续写入新事件和未读数。
    assert!(Arc::ptr_eq(&task, &fixture.tasks.task(&task.id)));
    fixture
        .repository
        .increment_unread_message_count(&session.id)
        .await?;
    queue_event(&task.output, reply("断线后产生的消息")).await?;
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(
        fixture
            .repository
            .get_by_id(&session.id)
            .await?
            .unwrap()
            .unread_message_count,
        1
    );
    assert_eq!(task.output.0.lock().unwrap().reads.len(), reads_before_drop);
    assert_eq!(task.output.size().await?, 2);
    assert!(!task.done());
    Ok(())
}

#[tokio::test]
async fn malformed_output_and_queue_failures_become_persisted_error_events() -> Result<()> {
    let fixture = Fixture::new().await?;
    for failure in ["input", "output", "decode"] {
        let (session, task) = fixture.subscription().await?;
        match failure {
            "input" => task.input.0.lock().unwrap().fail_put = true,
            "output" => task.output.0.lock().unwrap().fail_get = true,
            _ => {
                task.output.put(json!({"type": "future_event"})).await?;
            }
        }
        let events = fixture
            .chat(&session, (failure == "input").then_some("新消息"))
            .await;
        let message = error_message(&events);
        match failure {
            "input" => assert_eq!(message, "模拟队列写入失败"),
            "output" => assert_eq!(message, "模拟队列读取失败"),
            _ => assert!(message.contains("future_event")),
        }
        let saved = fixture.repository.get_by_id(&session.id).await?.unwrap();
        assert_eq!(saved.events, events);
        assert_eq!(saved.unread_message_count, 0);
        assert_eq!(task.invokes.load(Ordering::SeqCst), 0);
    }
    Ok(())
}

#[tokio::test]
async fn final_unread_reset_failure_keeps_the_original_queue_error() -> Result<()> {
    let fixture = Fixture::new().await?;
    let (session, task) = fixture.subscription().await?;
    // 仅在本测试私有数据库中注入清零失败，错误事件的追加仍然可以成功。
    fixture.database.db.execute_unprepared(
        "ALTER TABLE sessions ADD CONSTRAINT test_unread_nonzero CHECK (unread_message_count > 0)"
    ).await?;
    task.output.0.lock().unwrap().fail_get = true;
    let events = fixture.chat(&session, None).await;
    assert_eq!(error_message(&events), "模拟队列读取失败");
    let saved = fixture.repository.get_by_id(&session.id).await?.unwrap();
    assert_eq!(saved.events, events);
    assert_eq!(saved.unread_message_count, 5);
    Ok(())
}
