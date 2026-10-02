//! 会话沙箱读取测试：私有 PostgreSQL + 记录调用的沙箱，验证读取顺序与只读边界。

#[path = "support/file_database.rs"]
mod file_database;

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{anyhow, ensure, Result};
use async_trait::async_trait;
use migration::{Migrator, MigratorTrait, SchemaManager};
use sea_orm::{ConnectionTrait, DbBackend, Statement};
use server::{
    application::services::session_service::{SessionSandboxError, SessionService},
    domain::{
        external::{Browser, Sandbox, SandboxFactory},
        models::{Event, File, Session, SessionStatus, TitleEvent, ToolResult},
        repositories::SessionRepository,
    },
    infrastructure::repositories::SeaOrmSessionRepository,
};
use tokio::{sync::Notify, time::timeout};

#[derive(Debug, PartialEq, Eq)]
enum Call {
    Lookup(String),
    File(String),
    Shell(String),
    Vnc,
}

#[derive(Clone, Copy)]
enum ReadKind {
    File,
    Shell,
    Vnc,
}

impl ReadKind {
    async fn read(self, service: &SessionService, id: &str) -> Result<String> {
        match self {
            Self::File => service.read_file(id, "/home/ubuntu/测试文件.txt").await,
            Self::Shell => service.read_shell_output(id, "shell-session-42").await,
            Self::Vnc => service.get_vnc_url(id).await,
        }
    }

    fn call(self) -> Call {
        match self {
            Self::File => Call::File("/home/ubuntu/测试文件.txt".into()),
            Self::Shell => Call::Shell("shell-session-42".into()),
            Self::Vnc => Call::Vnc,
        }
    }
}

#[derive(Default)]
struct ReadGate {
    entered: Notify,
    release: Notify,
}

struct SandboxState {
    calls: Mutex<Vec<Call>>,
    available: Mutex<Result<bool, String>>,
    response: Mutex<Result<ToolResult<String>, String>>,
    gate: Mutex<Option<Arc<ReadGate>>>,
}

impl Default for SandboxState {
    fn default() -> Self {
        Self {
            calls: Mutex::default(),
            available: Mutex::new(Ok(true)),
            response: Mutex::new(Ok(ToolResult {
                data: Some(String::new()),
                ..ToolResult::default()
            })),
            gate: Mutex::default(),
        }
    }
}

impl SandboxState {
    async fn read(&self, call: Call) -> Result<ToolResult<String>> {
        self.calls.lock().unwrap().push(call);
        let gate = self.gate.lock().unwrap().clone();
        if let Some(gate) = gate {
            gate.entered.notify_one();
            gate.release.notified().await;
        }
        self.response
            .lock()
            .unwrap()
            .clone()
            .map_err(|message| anyhow!(message))
    }

    fn take_calls(&self) -> Vec<Call> {
        std::mem::take(&mut *self.calls.lock().unwrap())
    }
}

struct RecordingSandboxFactory(Arc<SandboxState>);

#[async_trait]
impl SandboxFactory for RecordingSandboxFactory {
    async fn create(&self) -> Result<Box<dyn Sandbox>> {
        panic!("查看已有内容只允许查找会话关联沙箱");
    }

    async fn get(&self, id: &str) -> Result<Option<Box<dyn Sandbox>>> {
        self.0.calls.lock().unwrap().push(Call::Lookup(id.into()));
        let available = self
            .0
            .available
            .lock()
            .unwrap()
            .clone()
            .map_err(|message| anyhow!(message))?;
        Ok(available.then(|| Box::new(RecordingSandbox(self.0.clone())) as Box<dyn Sandbox>))
    }
}

struct RecordingSandbox(Arc<SandboxState>);

#[async_trait]
impl Sandbox for RecordingSandbox {
    async fn read_file(
        &self,
        filepath: &str,
        start: Option<usize>,
        end: Option<usize>,
        sudo: Option<bool>,
        max_length: Option<usize>,
    ) -> Result<ToolResult<String>> {
        assert_eq!((start, end, sudo, max_length), (None, None, None, None));
        self.0.read(Call::File(filepath.into())).await
    }

    async fn read_shell_output(
        &self,
        session_id: &str,
        console: Option<bool>,
    ) -> Result<ToolResult<String>> {
        assert_eq!(console, Some(true));
        self.0.read(Call::Shell(session_id.into())).await
    }

    async fn exec_command(&self, _: &str, _: &str, _: &str) -> Result<ToolResult<String>> {
        panic!("读取不能执行命令")
    }
    async fn wait_process(&self, _: &str, _: Option<usize>) -> Result<ToolResult<String>> {
        panic!("读取不能等待进程")
    }
    async fn write_shell_input(
        &self,
        _: &str,
        _: &str,
        _: Option<bool>,
    ) -> Result<ToolResult<String>> {
        panic!("读取不能写入进程")
    }
    async fn kill_process(&self, _: &str) -> Result<ToolResult<String>> {
        panic!("读取不能终止进程")
    }
    async fn write_file(
        &self,
        _: &str,
        _: &str,
        _: Option<bool>,
        _: Option<bool>,
        _: Option<bool>,
        _: Option<bool>,
    ) -> Result<ToolResult<String>> {
        panic!("读取不能写入文件")
    }
    async fn check_file_exists(&self, _: &str) -> Result<ToolResult<bool>> {
        panic!("直接读取指定文件")
    }
    async fn delete_file(&self, _: &str) -> Result<ToolResult<String>> {
        panic!("读取不能删除文件")
    }
    async fn replace_in_file(
        &self,
        _: &str,
        _: &str,
        _: &str,
        _: Option<bool>,
    ) -> Result<ToolResult<String>> {
        panic!("读取不能替换文件")
    }
    async fn search_in_file(
        &self,
        _: &str,
        _: &str,
        _: Option<bool>,
    ) -> Result<ToolResult<Vec<String>>> {
        panic!("直接读取指定文件")
    }
    async fn find_files(&self, _: &str, _: &str) -> Result<ToolResult<Vec<String>>> {
        panic!("直接读取指定文件")
    }
    async fn upload_file(
        &self,
        _: Vec<u8>,
        _: &str,
        _: Option<&str>,
    ) -> Result<ToolResult<String>> {
        panic!("读取不能上传文件")
    }
    async fn download_file(&self, _: &str) -> Result<Vec<u8>> {
        panic!("直接读取文件内容")
    }
    async fn ensure_sandbox(&self) -> Result<bool> {
        panic!("读取不能初始化沙箱")
    }
    async fn destroy(&self) -> Result<bool> {
        panic!("读取不能销毁沙箱")
    }
    async fn get_browser(&self) -> Result<Box<dyn Browser>> {
        panic!("读取不能创建浏览器")
    }
    fn id(&self) -> &str {
        "关联沙箱"
    }
    fn cdp_url(&self) -> &str {
        panic!("读取不使用浏览器")
    }
    fn vnc_url(&self) -> &str {
        self.0.calls.lock().unwrap().push(Call::Vnc);
        "ws://sandbox.example:5901/websockify"
    }
}

struct Fixture {
    database: file_database::TestDatabase,
    repository: Arc<SeaOrmSessionRepository>,
    sandbox: Arc<SandboxState>,
    service: Arc<SessionService>,
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
        let sandbox = Arc::new(SandboxState::default());
        let service = Arc::new(SessionService::new(
            repository.clone(),
            Arc::new(RecordingSandboxFactory(sandbox.clone())),
        ));
        Ok(Self {
            database,
            repository,
            sandbox,
            service,
        })
    }

    async fn session(&self, sandbox_id: Option<&str>) -> Result<Session> {
        let session = Session {
            sandbox_id: sandbox_id.map(str::to_owned),
            task_id: Some("保留任务".into()),
            title: "查看已有执行内容".into(),
            unread_message_count: 7,
            latest_message: "保留最新消息".into(),
            status: SessionStatus::Waiting,
            events: vec![Event::Title(TitleEvent {
                title: "保留事件".into(),
                ..TitleEvent::default()
            })],
            files: vec![File {
                filename: "附件.txt".into(),
                size: 42,
                ..File::default()
            }],
            ..Session::default()
        };
        self.repository.save(session.clone()).await?;
        // 使用落库后精度作为基准，避免时间戳的纳秒/微秒转换影响只读断言。
        Ok(self.repository.get_by_id(&session.id).await?.unwrap())
    }
}

#[tokio::test]
async fn reads_only_the_associated_sandbox_and_preserves_session_history() -> Result<()> {
    let fixture = Fixture::new().await?;
    let session = fixture.session(Some("关联沙箱")).await?;
    for kind in [ReadKind::File, ReadKind::Shell, ReadKind::Vnc] {
        let contents = match kind {
            ReadKind::File => vec!["", "中文文件\n第二行\n"],
            ReadKind::Shell => vec![
                r#"{"output":"","session_id":"shell-session-42","console_records":[]}"#,
                r#"{"output":"执行成功\n","session_id":"shell-session-42","console_records":[{"command":"echo 测试","output":"测试","ps1":"ubuntu $"}]}"#,
            ],
            ReadKind::Vnc => vec!["ws://sandbox.example:5901/websockify"],
        };
        for content in contents {
            *fixture.sandbox.response.lock().unwrap() = Ok(ToolResult {
                data: Some(content.into()),
                ..ToolResult::default()
            });
            assert_eq!(kind.read(&fixture.service, &session.id).await?, content);
            assert_eq!(
                fixture.sandbox.take_calls(),
                vec![Call::Lookup("关联沙箱".into()), kind.call()]
            );
            assert_eq!(
                fixture.repository.get_by_id(&session.id).await?,
                Some(session.clone())
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn rejects_missing_sessions_and_unassigned_sandboxes_before_lookup() -> Result<()> {
    let fixture = Fixture::new().await?;
    for kind in [ReadKind::File, ReadKind::Shell, ReadKind::Vnc] {
        let id = uuid::Uuid::new_v4().to_string();
        let error = kind.read(&fixture.service, &id).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("当前会话不存在[{id}], 请核实后重试")
        );
        assert!(error.downcast_ref::<SessionSandboxError>().is_none());
        assert!(fixture.sandbox.take_calls().is_empty());

        for sandbox_id in [None, Some("")] {
            let session = fixture.session(sandbox_id).await?;
            let error = kind.read(&fixture.service, &session.id).await.unwrap_err();
            assert!(matches!(
                error.downcast_ref::<SessionSandboxError>(),
                Some(SessionSandboxError::Unassigned)
            ));
            assert!(fixture.sandbox.take_calls().is_empty());
            assert_eq!(
                fixture.repository.get_by_id(&session.id).await?,
                Some(session)
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn propagates_lookup_and_tool_failures_without_mutating_sessions() -> Result<()> {
    let fixture = Fixture::new().await?;
    let session = fixture.session(Some("关联沙箱")).await?;
    for kind in [ReadKind::File, ReadKind::Shell, ReadKind::Vnc] {
        *fixture.sandbox.available.lock().unwrap() = Ok(false);
        let error = kind.read(&fixture.service, &session.id).await.unwrap_err();
        assert!(matches!(
            error.downcast_ref::<SessionSandboxError>(),
            Some(SessionSandboxError::Unavailable)
        ));
        assert_eq!(
            fixture.sandbox.take_calls(),
            vec![Call::Lookup("关联沙箱".into())]
        );

        *fixture.sandbox.available.lock().unwrap() = Err("沙箱查找连接失败".into());
        let error = kind.read(&fixture.service, &session.id).await.unwrap_err();
        assert_eq!(error.to_string(), "沙箱查找连接失败");
        assert!(error.downcast_ref::<SessionSandboxError>().is_none());
        assert_eq!(
            fixture.sandbox.take_calls(),
            vec![Call::Lookup("关联沙箱".into())]
        );

        // 获取 VNC 地址只检查会话与沙箱；文件和 Shell 才继续调用读取工具。
        if matches!(kind, ReadKind::Vnc) {
            assert_eq!(
                fixture.repository.get_by_id(&session.id).await?,
                Some(session.clone())
            );
            continue;
        }

        *fixture.sandbox.available.lock().unwrap() = Ok(true);
        *fixture.sandbox.response.lock().unwrap() =
            Ok(ToolResult::from_sandbox(500, "文件或 Shell 不存在", None));
        let error = kind.read(&fixture.service, &session.id).await.unwrap_err();
        assert!(
            matches!(error.downcast_ref::<SessionSandboxError>(), Some(SessionSandboxError::RequestFailed(message)) if message == "文件或 Shell 不存在")
        );
        assert_eq!(
            fixture.sandbox.take_calls(),
            vec![Call::Lookup("关联沙箱".into()), kind.call()]
        );

        *fixture.sandbox.response.lock().unwrap() = Err("沙箱读取网络中断".into());
        let error = kind.read(&fixture.service, &session.id).await.unwrap_err();
        assert_eq!(error.to_string(), "沙箱读取网络中断");
        assert!(error.downcast_ref::<SessionSandboxError>().is_none());
        assert_eq!(
            fixture.sandbox.take_calls(),
            vec![Call::Lookup("关联沙箱".into()), kind.call()]
        );

        *fixture.sandbox.response.lock().unwrap() = Ok(ToolResult::default());
        let error = kind.read(&fixture.service, &session.id).await.unwrap_err();
        assert!(
            error.downcast_ref::<SessionSandboxError>().is_none(),
            "成功响应缺少 data 应保留为内部错误"
        );
        assert_eq!(
            fixture.sandbox.take_calls(),
            vec![Call::Lookup("关联沙箱".into()), kind.call()]
        );
        assert_eq!(
            fixture.repository.get_by_id(&session.id).await?,
            Some(session.clone())
        );
    }
    Ok(())
}

#[tokio::test]
async fn stops_on_database_failure_before_accessing_the_sandbox() -> Result<()> {
    let fixture = Fixture::new().await?;
    let session = fixture.session(Some("关联沙箱")).await?;
    // 删除的仅为此测试创建的私有数据库表，以覆盖真实仓库错误的传播。
    fixture
        .database
        .db
        .execute_unprepared("DROP TABLE sessions")
        .await?;
    for kind in [ReadKind::File, ReadKind::Shell, ReadKind::Vnc] {
        assert!(kind.read(&fixture.service, &session.id).await.is_err());
        assert!(fixture.sandbox.take_calls().is_empty());
    }
    Ok(())
}

#[tokio::test]
async fn sandbox_waits_leave_database_transactions_closed() -> Result<()> {
    let fixture = Fixture::new().await?;
    let session = fixture.session(Some("关联沙箱")).await?;
    for kind in [ReadKind::File, ReadKind::Shell] {
        let gate = Arc::new(ReadGate::default());
        *fixture.sandbox.gate.lock().unwrap() = Some(gate.clone());
        let service = fixture.service.clone();
        let id = session.id.clone();
        let read = tokio::spawn(async move { kind.read(&service, &id).await });
        timeout(Duration::from_secs(5), gate.entered.notified()).await?;

        // 沙箱仍等待时检查数据库，确保网络等待阶段没有占用未提交事务。
        let row = timeout(Duration::from_secs(5), fixture.database.db.query_one_raw(Statement::from_string(
            DbBackend::Postgres,
            "SELECT count(*)::bigint AS active_transactions FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid() AND xact_start IS NOT NULL",
        ))).await??.unwrap();
        assert_eq!(row.try_get::<i64>("", "active_transactions")?, 0);
        assert_eq!(
            timeout(
                Duration::from_secs(5),
                fixture.repository.get_by_id(&session.id)
            )
            .await??,
            Some(session.clone())
        );

        gate.release.notify_one();
        timeout(Duration::from_secs(5), read).await???;
        assert_eq!(
            fixture.sandbox.take_calls(),
            vec![Call::Lookup("关联沙箱".into()), kind.call()]
        );
    }
    Ok(())
}
