//! 会话仓库的真实 PostgreSQL 集成测试。
//!
//! 运行：`cargo test --test session_repository`。
//! 测试自行创建临时数据库实例，只监听私有 Unix socket，不使用应用配置或现有数据库。
//! PostgreSQL 的程序需要在 PATH 中，或通过 SESSION_REPOSITORY_PG_BIN 指定目录。

use std::{
    collections::HashSet,
    env,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::Arc,
    time::Duration,
};

use anyhow::{bail, ensure, Context, Result};
use chrono::{TimeZone, Utc};
use futures::future::try_join_all;
use migration::{Migrator, MigratorTrait, SchemaManager};
use sea_orm::{
    ColumnTrait, ConnectOptions, ConnectionTrait, Database, DatabaseBackend, DatabaseConnection,
    EntityTrait, PaginatorTrait, QueryFilter, Statement,
};
use serde_json::json;
use server::{
    domain::{
        models::{
            DoneEvent, ErrorEvent, Event, File, McpToolContent, Memory, MessageEvent, MessageRole,
            Plan, PlanEvent, Session, SessionStatus, Step, StepEvent, TitleEvent, ToolContent,
            ToolEvent, ToolEventStatus, ToolResult, WaitEvent,
        },
        repositories::SessionRepository,
    },
    infrastructure::repositories::SeaOrmSessionRepository,
    models::sessions::{Column, Entity, Model},
};
use tempfile::TempDir;
use tokio::sync::Barrier;
use uuid::Uuid;

/// 每个测试使用独立实例；即使断言 panic，Drop 也会停止该实例并移除临时目录。
struct TestDatabase {
    db: DatabaseConnection,
    _postgres: TemporaryPostgres,
}

impl TestDatabase {
    async fn new() -> Result<Self> {
        let postgres = TemporaryPostgres::start()?;
        let mut options = ConnectOptions::new(format!(
            "postgres://postgres@localhost/postgres?host={}",
            postgres.directory.path().display()
        ));
        options
            .min_connections(1)
            .max_connections(8)
            .connect_timeout(Duration::from_secs(10))
            .sqlx_logging(false);
        let db = Database::connect(options)
            .await
            .context("无法连接测试自行创建的 PostgreSQL 实例")?;
        let fixture = Self {
            db,
            _postgres: postgres,
        };

        // 直接复用课程的建表和索引迁移，避免测试手写另一套表结构。
        let manager = SchemaManager::new(&fixture.db);
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

        Ok(fixture)
    }

    fn repository(&self) -> SeaOrmSessionRepository {
        SeaOrmSessionRepository::new(self.db.clone())
    }

    async fn row(&self, session_id: &str) -> Result<Model> {
        Entity::find()
            .filter(Column::Uuid.eq(Uuid::parse_str(session_id)?))
            .one(&self.db)
            .await?
            .context("测试期望会话记录存在")
    }

    async fn execute(&self, sql: &str, values: Vec<sea_orm::Value>) -> Result<()> {
        self.db
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                sql,
                values,
            ))
            .await?;
        Ok(())
    }
}

struct TemporaryPostgres {
    directory: TempDir,
    bin_dir: PathBuf,
}

impl TemporaryPostgres {
    fn start() -> Result<Self> {
        let bin_dir = postgres_bin_dir()?;
        // 使用短路径以避免 macOS 的 Unix socket 路径长度限制。
        let directory = tempfile::Builder::new()
            .prefix("session-repository-")
            .tempdir_in("/tmp")?;
        let data_dir = directory.path().join("data");
        command_output(
            Command::new(bin_dir.join("initdb"))
                .arg("-D")
                .arg(&data_dir)
                .args([
                    "--username=postgres",
                    "--auth=trust",
                    "--no-sync",
                    "--encoding=UTF8",
                    "--locale=C",
                ]),
        )?;

        let instance = Self { directory, bin_dir };
        command_output(
            Command::new(instance.bin_dir.join("pg_ctl"))
                .arg("-D")
                .arg(&data_dir)
                .arg("-l")
                .arg(instance.directory.path().join("postgres.log"))
                .args(["-w", "-t", "15", "-o"])
                .arg(format!(
                    "-c listen_addresses='' -c unix_socket_directories='{}' \
                     -c shared_buffers=8MB -c max_connections=16 -c fsync=off",
                    instance.directory.path().display()
                ))
                .arg("start"),
        )?;
        Ok(instance)
    }
}

impl Drop for TemporaryPostgres {
    fn drop(&mut self) {
        // 只停止本夹具 initdb 出来的数据目录，绝不操作系统已有 PostgreSQL 服务。
        let stopped = Command::new(self.bin_dir.join("pg_ctl"))
            .arg("-D")
            .arg(self.directory.path().join("data"))
            .args(["-m", "immediate", "-w", "-t", "10", "stop"])
            .output();
        if !matches!(stopped, Ok(ref output) if output.status.success()) {
            eprintln!("临时 PostgreSQL 停止失败：{stopped:?}");
        }
    }
}

fn postgres_bin_dir() -> Result<PathBuf> {
    if let Some(directory) = env::var_os("SESSION_REPOSITORY_PG_BIN") {
        let directory = PathBuf::from(directory);
        ensure!(
            directory.join("initdb").is_file() && directory.join("pg_ctl").is_file(),
            "SESSION_REPOSITORY_PG_BIN 必须指向包含 initdb 和 pg_ctl 的目录"
        );
        return Ok(directory);
    }

    let path = env::var_os("PATH").unwrap_or_default();
    env::split_paths(&path)
        .chain([
            Path::new("/usr/local/opt/postgresql@17/bin").to_path_buf(),
            Path::new("/opt/homebrew/opt/postgresql@17/bin").to_path_buf(),
        ])
        .find(|directory| directory.join("initdb").is_file() && directory.join("pg_ctl").is_file())
        .context("集成测试需要 PostgreSQL；请安装并设置 SESSION_REPOSITORY_PG_BIN，测试不会跳过")
}

fn command_output(command: &mut Command) -> Result<Output> {
    let output = command
        .output()
        .with_context(|| format!("无法执行测试数据库程序：{command:?}"))?;
    if !output.status.success() {
        bail!(
            "测试数据库程序执行失败：{command:?}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output)
}

fn memory(content: &str) -> Memory {
    Memory {
        messages: vec![json!({"role": "assistant", "content": content})
            .as_object()
            .unwrap()
            .clone()],
    }
}

fn file(filename: &str) -> File {
    File {
        filename: filename.to_string(),
        filepath: format!("/workspace/{filename}"),
        key: format!("test/{filename}"),
        extension: "md".to_string(),
        mime_type: "text/markdown".to_string(),
        size: 42,
        ..File::default()
    }
}

#[tokio::test]
async fn saves_by_uuid_without_overwriting_database_metadata() -> Result<()> {
    let fixture = TestDatabase::new().await?;
    let repository = fixture.repository();
    let mut session = Session {
        title: "初始标题".to_string(),
        ..Session::default()
    };
    repository.save(session.clone()).await?;
    let user_id = Uuid::new_v4();
    fixture
        .execute(
            "UPDATE sessions SET user_id = $1, remark = $2, is_deleted = true, \
             updated_at = '2000-01-01T00:00:00Z' WHERE uuid = $3",
            vec![
                user_id.into(),
                "数据库独有备注".into(),
                Uuid::parse_str(&session.id)?.into(),
            ],
        )
        .await?;
    let before = fixture.row(&session.id).await?;

    session.title = "更新标题".to_string();
    session.sandbox_id = Some("sandbox-1".to_string());
    session.task_id = Some("task-1".to_string());
    session.unread_message_count = 3;
    session.latest_message = "最新消息".to_string();
    session.latest_message_at = Some(Utc.with_ymd_and_hms(2026, 9, 1, 12, 0, 0).unwrap());
    session.status = SessionStatus::Running;
    session.events.push(Event::Message(MessageEvent {
        message: "完整消息".to_string(),
        ..MessageEvent::default()
    }));
    session.files.push(file("result.md"));
    session
        .memories
        .insert("planner".to_string(), memory("分析"));
    // 领域传入时间不应改写数据库既有的创建时间。
    session.created_at = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).unwrap();
    repository.save(session.clone()).await?;

    let after = fixture.row(&session.id).await?;
    assert_eq!(after.id, before.id);
    assert_eq!(after.created_at, before.created_at);
    assert!(after.updated_at > before.updated_at);
    assert_eq!(after.user_id, Some(user_id));
    assert_eq!(after.remark.as_deref(), Some("数据库独有备注"));
    assert_eq!(after.is_deleted, Some(true));
    let saved = repository.get_by_id(&session.id).await?.unwrap();
    // 时间由数据库管理，其他领域字段应完整往返。
    session.created_at = saved.created_at;
    session.updated_at = saved.updated_at;
    assert_eq!(saved, session);

    // 多个请求同时创建相同 UUID 时也只能产生一行，不能先查再插导致冲突。
    let concurrent_session = Session::default();
    let barrier = Barrier::new(12);
    try_join_all((0..12).map(|_| {
        let session = concurrent_session.clone();
        let repository = &repository;
        let barrier = &barrier;
        async move {
            barrier.wait().await;
            repository.save(session).await
        }
    }))
    .await?;
    assert_eq!(
        Entity::find()
            .filter(Column::Uuid.eq(Uuid::parse_str(&concurrent_session.id)?))
            .count(&fixture.db)
            .await?,
        1
    );
    Ok(())
}

#[tokio::test]
async fn reads_orders_updates_and_deletes_sessions() -> Result<()> {
    let fixture = TestDatabase::new().await?;
    let repository = fixture.repository();
    assert!(repository.get_all().await?.is_empty());

    let no_message = Session::default();
    let earlier = Session {
        latest_message_at: Some(Utc.with_ymd_and_hms(2026, 9, 1, 12, 0, 0).unwrap()),
        ..Session::default()
    };
    let later = Session::default();
    for session in [&no_message, &later, &earlier] {
        repository.save(session.clone()).await?;
    }
    fixture
        .execute(
            "UPDATE sessions SET updated_at = '2000-01-01T00:00:00Z' WHERE uuid = $1",
            vec![Uuid::parse_str(&later.id)?.into()],
        )
        .await?;
    let before = fixture.row(&later.id).await?.updated_at;
    let latest_at = Utc.with_ymd_and_hms(2026, 9, 2, 12, 0, 0).unwrap();
    repository.update_title(&later.id, "新的标题").await?;
    assert!(fixture.row(&later.id).await?.updated_at > before);
    repository
        .update_latest_message(&later.id, "新的回复", latest_at)
        .await?;
    repository
        .update_status(&later.id, SessionStatus::Completed)
        .await?;

    let saved = repository.get_by_id(&later.id).await?.unwrap();
    assert_eq!(saved.title, "新的标题");
    assert_eq!(saved.latest_message, "新的回复");
    assert_eq!(saved.latest_message_at, Some(latest_at));
    assert_eq!(saved.status, SessionStatus::Completed);
    assert_eq!(
        repository
            .get_all()
            .await?
            .into_iter()
            .map(|session| session.id)
            .collect::<Vec<_>>(),
        vec![later.id.clone(), earlier.id, no_message.id]
    );

    repository.delete_by_id(&later.id).await?;
    repository.delete_by_id(&later.id).await?;
    assert!(repository.get_by_id(&later.id).await?.is_none());
    assert_eq!(repository.get_all().await?.len(), 2);
    Ok(())
}

#[tokio::test]
async fn missing_sessions_have_explicit_query_and_update_contracts() -> Result<()> {
    let fixture = TestDatabase::new().await?;
    let repository = fixture.repository();
    let missing = Uuid::new_v4().to_string();

    assert!(repository.get_by_id(&missing).await?.is_none());
    assert!(repository
        .get_file_by_path(&missing, "/workspace/missing.md")
        .await?
        .is_none());
    repository.delete_by_id(&missing).await?;
    assert!(repository.update_title(&missing, "标题").await.is_err());
    assert!(repository
        .update_latest_message(&missing, "消息", Utc::now())
        .await
        .is_err());
    assert!(repository
        .update_unread_message_count(&missing, 0)
        .await
        .is_err());
    assert!(repository
        .increment_unread_message_count(&missing)
        .await
        .is_err());
    assert!(repository
        .decrement_unread_message_count(&missing)
        .await
        .is_err());
    assert!(repository
        .update_status(&missing, SessionStatus::Running)
        .await
        .is_err());
    assert!(repository
        .add_event(&missing, Event::Done(DoneEvent::default()))
        .await
        .is_err());
    assert!(repository
        .add_file(&missing, file("missing.md"))
        .await
        .is_err());
    assert!(repository.remove_file(&missing, "file-id").await.is_err());
    assert!(repository
        .save_memory(&missing, "planner", memory("计划"))
        .await
        .is_err());
    assert!(repository.get_memory(&missing, "planner").await.is_err());
    assert!(repository.get_by_id("invalid-uuid").await.is_err());
    Ok(())
}

#[tokio::test]
async fn unread_counts_handle_null_zero_and_database_integer_bounds() -> Result<()> {
    let fixture = TestDatabase::new().await?;
    let repository = fixture.repository();
    let session = Session::default();
    repository.save(session.clone()).await?;

    fixture
        .execute(
            "UPDATE sessions SET unread_message_count = NULL WHERE uuid = $1",
            vec![Uuid::parse_str(&session.id)?.into()],
        )
        .await?;
    assert_eq!(
        repository
            .get_by_id(&session.id)
            .await?
            .unwrap()
            .unread_message_count,
        0
    );
    repository
        .increment_unread_message_count(&session.id)
        .await?;
    assert_eq!(
        fixture.row(&session.id).await?.unread_message_count,
        Some(1)
    );
    repository
        .decrement_unread_message_count(&session.id)
        .await?;
    repository
        .decrement_unread_message_count(&session.id)
        .await?;
    assert_eq!(
        fixture.row(&session.id).await?.unread_message_count,
        Some(0)
    );

    fixture
        .execute(
            "UPDATE sessions SET unread_message_count = NULL WHERE uuid = $1",
            vec![Uuid::parse_str(&session.id)?.into()],
        )
        .await?;
    repository
        .decrement_unread_message_count(&session.id)
        .await?;
    assert_eq!(
        fixture.row(&session.id).await?.unread_message_count,
        Some(0)
    );

    repository
        .update_unread_message_count(&session.id, i32::MAX as usize)
        .await?;
    let before = fixture.row(&session.id).await?;
    assert!(repository
        .increment_unread_message_count(&session.id)
        .await
        .is_err());
    assert!(repository
        .update_unread_message_count(&session.id, i32::MAX as usize + 1)
        .await
        .is_err());
    assert!(repository
        .save(Session {
            unread_message_count: usize::MAX,
            ..session.clone()
        })
        .await
        .is_err());
    assert_eq!(fixture.row(&session.id).await?, before);
    Ok(())
}

#[tokio::test]
async fn json_collections_and_all_event_variants_round_trip() -> Result<()> {
    let fixture = TestDatabase::new().await?;
    let repository = fixture.repository();
    let session = Session::default();
    repository.save(session.clone()).await?;
    // 历史记录可能为 SQL NULL；第一次写入时需要建立数组或对象。
    fixture
        .execute(
            "UPDATE sessions SET events = NULL, files = NULL, memories = NULL WHERE uuid = $1",
            vec![Uuid::parse_str(&session.id)?.into()],
        )
        .await?;
    assert_eq!(
        repository.get_memory(&session.id, "planner").await?,
        Memory::new()
    );
    let attachment = file("用户's笔记.md");
    let events = vec![
        Event::Plan(PlanEvent {
            plan: Plan {
                steps: vec![Step::new("查找资料")],
                ..Plan::new("研究计划", "完成报告")
            },
            ..PlanEvent::default()
        }),
        Event::Title(TitleEvent {
            title: "报告会话".to_string(),
            ..TitleEvent::default()
        }),
        Event::Step(StepEvent {
            step: Step::new("编写报告"),
            ..StepEvent::default()
        }),
        Event::Message(MessageEvent {
            role: MessageRole::User,
            message: "请读取附件".to_string(),
            attachments: vec![attachment.clone()],
            ..MessageEvent::default()
        }),
        Event::Tool(ToolEvent {
            tool_call_id: "call-1".to_string(),
            tool_name: "filesystem".to_string(),
            function_name: "read_file".to_string(),
            function_args: json!({"path": attachment.filepath})
                .as_object()
                .unwrap()
                .clone(),
            function_result: Some(ToolResult::from_sandbox(
                200,
                "成功",
                Some(json!({"content": "文件正文", "lines": [1, 2]})),
            )),
            tool_content: Some(ToolContent::Mcp(McpToolContent {
                result: json!({"source": "filesystem"}),
            })),
            status: ToolEventStatus::Called,
            ..ToolEvent::default()
        }),
        Event::Wait(WaitEvent::default()),
        Event::Error(ErrorEvent {
            error: "文件未找到".to_string(),
            ..ErrorEvent::default()
        }),
        Event::Done(DoneEvent::default()),
    ];
    for event in &events {
        repository.add_event(&session.id, event.clone()).await?;
    }
    assert_eq!(
        repository.get_by_id(&session.id).await?.unwrap().events,
        events
    );

    let second = file("report.md");
    repository.add_file(&session.id, attachment.clone()).await?;
    repository.add_file(&session.id, second.clone()).await?;
    repository.add_file(&session.id, attachment.clone()).await?;
    assert_eq!(
        repository
            .get_file_by_path(&session.id, &attachment.filepath)
            .await?,
        Some(attachment.clone())
    );
    assert!(repository
        .get_file_by_path(&session.id, "missing")
        .await?
        .is_none());
    repository.remove_file(&session.id, "missing-id").await?;
    repository.remove_file(&session.id, &attachment.id).await?;
    repository.remove_file(&session.id, &attachment.id).await?;
    assert_eq!(
        repository.get_by_id(&session.id).await?.unwrap().files,
        vec![second]
    );

    // Agent 名是 JSON 对象的普通键，包含引号、点或括号也不能被解释成 SQL/JSON 路径。
    let planner = "planner.'{中文}";
    repository
        .save_memory(&session.id, planner, memory("旧计划"))
        .await?;
    repository
        .save_memory(&session.id, "executor", memory("执行历史"))
        .await?;
    repository
        .save_memory(&session.id, planner, memory("新计划"))
        .await?;
    assert_eq!(
        repository.get_memory(&session.id, planner).await?,
        memory("新计划")
    );
    assert_eq!(
        repository.get_memory(&session.id, "executor").await?,
        memory("执行历史")
    );
    assert_eq!(
        repository.get_memory(&session.id, "missing").await?,
        Memory::new()
    );
    assert_eq!(
        repository
            .get_by_id(&session.id)
            .await?
            .unwrap()
            .memories
            .len(),
        2
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_updates_keep_all_counters_events_files_and_agent_memories() -> Result<()> {
    let fixture = TestDatabase::new().await?;
    let repository = Arc::new(fixture.repository());
    let session = Session::default();
    repository.save(session.clone()).await?;
    let workers = 24;
    let barrier = Arc::new(Barrier::new(workers));

    try_join_all((0..workers).map(|index| {
        let repository = Arc::clone(&repository);
        let barrier = Arc::clone(&barrier);
        let session_id = session.id.clone();
        async move {
            barrier.wait().await;
            repository
                .increment_unread_message_count(&session_id)
                .await?;
            repository
                .add_event(
                    &session_id,
                    Event::Message(MessageEvent {
                        message: format!("message-{index}"),
                        ..MessageEvent::default()
                    }),
                )
                .await?;
            repository
                .save_memory(
                    &session_id,
                    &format!("agent-{index}"),
                    memory(&index.to_string()),
                )
                .await?;
            repository
                .add_file(&session_id, file(&format!("file-{index}.md")))
                .await?;
            Ok::<_, anyhow::Error>(())
        }
    }))
    .await?;

    let saved = repository.get_by_id(&session.id).await?.unwrap();
    assert_eq!(saved.unread_message_count, workers);
    assert_eq!(saved.events.len(), workers);
    assert_eq!(saved.files.len(), workers);
    assert_eq!(saved.memories.len(), workers);
    let messages = saved
        .events
        .iter()
        .map(|event| match event {
            Event::Message(event) => event.message.clone(),
            event => panic!("事件往返后类型发生改变：{event:?}"),
        })
        .collect::<HashSet<_>>();
    let filenames = saved
        .files
        .iter()
        .map(|file| file.filename.clone())
        .collect::<HashSet<_>>();
    for index in 0..workers {
        assert!(messages.contains(&format!("message-{index}")));
        assert!(filenames.contains(&format!("file-{index}.md")));
        assert_eq!(
            saved.memories.get(&format!("agent-{index}")),
            Some(&memory(&index.to_string()))
        );
    }
    Ok(())
}
