//! 会话 API 集成测试：复用私有临时 PostgreSQL，绕过应用 boot 和现有数据库。

#[path = "support/file_database.rs"]
mod file_database;

use std::time::Duration;

use anyhow::{ensure, Context, Result};
use axum::Router;
use axum_test::TestServer;
use chrono::{TimeZone, Utc};
use loco_rs::{
    app::{AppContext, Hooks},
    config::Config,
    environment::Environment,
};
use migration::{Migrator, MigratorTrait, SchemaManager};
use sea_orm::ConnectionTrait;
use serde_json::{json, Value};
use serial_test::serial;
use server::{
    app::App,
    domain::{
        models::{
            DoneEvent, ErrorEvent, Event, File, FileToolContent, MessageEvent, PlanEvent, Session,
            SessionStatus, StepEvent, TitleEvent, ToolContent, ToolEvent, ToolEventStatus,
            ToolResult, WaitEvent,
        },
        repositories::SessionRepository,
    },
    infrastructure::repositories::SeaOrmSessionRepository,
    views::events::AgentSseEvent,
};
use tokio::time::{timeout, timeout_at, Instant};

struct TestApp {
    server: TestServer,
    database: file_database::TestDatabase,
}

impl TestApp {
    async fn new() -> Result<Self> {
        Self::with_agent_config(false).await
    }

    async fn with_agent_config(agent_configured: bool) -> Result<Self> {
        Self::with_transport(agent_configured, false).await
    }

    async fn with_transport(agent_configured: bool, http_transport: bool) -> Result<Self> {
        let database = file_database::TestDatabase::new().await?;
        // 复用已有临时实例，在它的私有数据库内执行实际会话表迁移。
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

        if agent_configured {
            // 配置表为空时服务使用默认配置；仅连接本测试的私有数据库。
            let migrations = Migrator::migrations()
                .into_iter()
                .filter(|migration| {
                    matches!(
                        migration.name(),
                        "m20260526_131658_llm_configs"
                            | "m20260526_134746_fix_llm_configs_table"
                            | "m20260601_143631_agent_configs"
                            | "m20260601_144016_fix_agent_configs_table"
                            | "m20260605_185020_mcp_servers"
                            | "m20260717_191151_a2a_servers"
                            | "m20260718_113716_fix_a2a_servers_table"
                    )
                })
                .collect::<Vec<_>>();
            ensure!(migrations.len() == 7, "没有找到配置表的七条迁移");
            for migration in migrations {
                migration.up(&manager).await?;
            }
        }

        let config: Config = serde_json::from_value(json!({
            "logger": { "enable": false, "level": "info", "format": "compact" },
            "server": { "port": 0, "host": "http://localhost" },
            // 查询已有任务和空消息不需要 Redis 网络连接，端口1也应正常完成。
            "cache": if agent_configured {
                json!({"kind": "Redis", "uri": "redis://127.0.0.1:1", "max_size": 1})
            } else { json!({"kind": "Null"}) },
            "database": {
                "uri": "unused", "enable_logging": false,
                "min_connections": 1, "max_connections": 1,
                "connect_timeout": 10, "idle_timeout": 10
            }
        }))?;
        let ctx = AppContext::builder(Environment::Test, database.db.clone(), config).build();
        // 通过真实应用路由覆盖依赖注入、控制器、服务及仓库的完整链路。
        let router = App::routes(&ctx).to_router::<App>(ctx, Router::new())?;
        let server = if http_transport {
            // 随机监听端口，客户端按帧读取持续推送的会话列表。
            TestServer::builder().http_transport().build(router)?
        } else {
            TestServer::new(router)?
        };
        Ok(Self { server, database })
    }

    fn repository(&self) -> SeaOrmSessionRepository {
        SeaOrmSessionRepository::new(self.database.db.clone())
    }
}

#[tokio::test]
#[serial]
async fn creates_blank_sessions_and_lists_only_basic_information() -> Result<()> {
    let app = TestApp::new().await?;
    let empty = app.server.get("/api/sessions").await;
    empty.assert_status_ok();
    assert_eq!(
        empty.json::<Value>(),
        json!({"code": 200, "msg": "获取任务会话列表成功", "data": {"sessions": []}})
    );

    let created = app.server.post("/api/sessions").await;
    created.assert_status_ok();
    let created: Value = created.json();
    let id = created["data"]["session_id"].as_str().unwrap();
    assert!(uuid::Uuid::parse_str(id).is_ok());
    assert_eq!(
        created,
        json!({"code": 200, "msg": "创建任务会话成功", "data": {"session_id": id}})
    );
    let stored = app.repository().get_by_id(id).await?.unwrap();
    assert_eq!(stored.title, "新对话");
    assert_eq!(stored.status, SessionStatus::Pending);
    assert_eq!(stored.unread_message_count, 0);
    assert!(stored.sandbox_id.is_none() && stored.task_id.is_none());
    assert!(stored.events.is_empty() && stored.files.is_empty() && stored.memories.is_empty());

    let list = app.server.get("/api/sessions").await;
    list.assert_status_ok();
    assert_eq!(
        list.json::<Value>()["data"]["sessions"],
        json!([{
            "session_id": id, "title": "新对话", "latest_message": "",
            "latest_message_at": null, "status": "pending", "unread_message_count": 0
        }])
    );

    let second = app.server.post("/api/sessions").await;
    second.assert_status_ok();
    let second: Value = second.json();
    let second_id = second["data"]["session_id"].as_str().unwrap();
    assert_ne!(id, second_id);
    let repository = app.repository();
    repository.update_title(second_id, "有消息的会话").await?;
    repository
        .update_latest_message(
            second_id,
            "任务进行中",
            Utc.with_ymd_and_hms(2026, 9, 30, 12, 0, 0).unwrap(),
        )
        .await?;
    repository
        .update_status(second_id, SessionStatus::Running)
        .await?;
    repository.update_unread_message_count(second_id, 3).await?;
    repository
        .add_event(second_id, Event::Title(TitleEvent::default()))
        .await?;
    // 有消息的会话在前；列表仍只包含六个公开字段，避免暴露完整事件。
    let list = app.server.get("/api/sessions").await;
    list.assert_status_ok();
    let list: Value = list.json();
    let items = list["data"]["sessions"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(
        items[0],
        json!({
            "session_id": second_id, "title": "有消息的会话", "latest_message": "任务进行中",
            "latest_message_at": "2026-09-30T12:00:00Z", "status": "running", "unread_message_count": 3
        })
    );
    assert_eq!(items[1]["session_id"], id);
    Ok(())
}

#[tokio::test]
#[serial]
async fn gets_session_details_with_ordered_events_without_changing_the_session() -> Result<()> {
    let app = TestApp::new().await?;
    let repository = app.repository();
    let session = Session {
        title: "已保存的会话历史".into(),
        unread_message_count: 7,
        status: SessionStatus::Waiting,
        sandbox_id: Some("保留沙箱".into()),
        task_id: Some("保留任务".into()),
        events: vec![
            Event::Title(TitleEvent {
                title: "读取文件".into(),
                ..TitleEvent::default()
            }),
            Event::Message(MessageEvent {
                message: "已经读取文件".into(),
                ..MessageEvent::default()
            }),
            Event::Plan(PlanEvent::default()),
            Event::Step(StepEvent::default()),
            Event::Tool(ToolEvent {
                tool_name: "file".into(),
                function_name: "file_read".into(),
                status: ToolEventStatus::Called,
                tool_content: Some(ToolContent::File(FileToolContent {
                    content: "调用时的文件快照".into(),
                })),
                function_result: Some(ToolResult {
                    data: Some(json!({"internal": "原始工具结果保留在数据库"})),
                    ..ToolResult::default()
                }),
                ..ToolEvent::default()
            }),
            Event::Wait(WaitEvent::default()),
            Event::Error(ErrorEvent::default()),
            Event::Done(DoneEvent::default()),
        ],
        ..Session::default()
    };
    repository.save(session.clone()).await?;
    let before = repository.get_by_id(&session.id).await?.unwrap();

    // Null cache 且没有模型配置：读取详情仅查询历史，不构建或启动 Agent 任务。
    let response = app
        .server
        .get(&format!("/api/sessions/{}", session.id))
        .await;
    response.assert_status_ok();
    let body: Value = response.json();
    assert_eq!(
        body,
        json!({
            "code": 200,
            "msg": "获取会话详情成功",
            "data": {
                "session_id": session.id,
                "title": session.title,
                "status": "waiting",
                "events": AgentSseEvent::from_events(before.events.clone())
            }
        })
    );
    assert_eq!(
        body["data"]["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|event| event["event"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["title", "message", "plan", "step", "tool", "wait", "error", "done"]
    );
    assert_eq!(
        body["data"]["events"][4]["data"]["content"],
        json!({"content": "调用时的文件快照"})
    );
    assert!(!response.text().contains("原始工具结果保留在数据库"));
    // 查看详情保留未读数及全部原始数据，复用聊天流的公开事件格式。
    assert_eq!(repository.get_by_id(&session.id).await?.unwrap(), before);

    let blank = Session::default();
    repository.save(blank.clone()).await?;
    let response = app.server.get(&format!("/api/sessions/{}", blank.id)).await;
    response.assert_status_ok();
    assert_eq!(response.json::<Value>()["data"]["events"], json!([]));

    let missing = app
        .server
        .get(&format!("/api/sessions/{}", uuid::Uuid::new_v4()))
        .await;
    missing.assert_status_not_found();
    assert_eq!(missing.json::<Value>()["error"], "session.not_found");
    assert_eq!(
        missing.json::<Value>()["description"],
        "该会话不存在，请核实后重试"
    );
    Ok(())
}

/// 以完整 SSE 帧为单位解码，避免 TCP 分块截断中文字符或无限等待整个响应。
async fn next_sessions_frame(
    response: &mut reqwest::Response,
    buffered: &mut Vec<u8>,
) -> Result<Value> {
    loop {
        if let Some(end) = buffered.windows(2).position(|bytes| bytes == b"\n\n") {
            let frame = String::from_utf8(buffered.drain(..end + 2).collect())?;
            let event = frame
                .lines()
                .find_map(|line| line.strip_prefix("event:"))
                .context("会话列表 SSE 缺少事件名")?;
            ensure!(event.trim() == "sessions", "会话列表事件名应为 sessions");
            let data = frame
                .lines()
                .find_map(|line| line.strip_prefix("data:"))
                .context("会话列表 SSE 缺少数据")?;
            return Ok(serde_json::from_str(data)?);
        }
        let chunk = response.chunk().await?.context("会话列表流提前结束")?;
        buffered.extend_from_slice(&chunk);
    }
}

#[tokio::test]
#[serial]
async fn streams_full_session_snapshots_immediately_then_every_five_seconds() -> Result<()> {
    let app = TestApp::with_transport(false, true).await?;
    let client = reqwest::Client::new();
    let url = app.server.server_url("/api/sessions/stream")?;
    // 流式列表接口不要求 JSON 请求体；先收到空列表，再等待下一次查询。
    let mut response = timeout(Duration::from_secs(3), client.post(url).send()).await??;
    assert!(response.status().is_success());
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let mut buffered = Vec::new();
    let first = timeout(
        Duration::from_secs(2),
        next_sessions_frame(&mut response, &mut buffered),
    )
    .await??;
    assert_eq!(first, json!({"sessions": []}));
    let first_received = Instant::now();

    // SSE 挂起期间，普通接口和数据库更新均能正常完成。
    let created = timeout(Duration::from_secs(2), async {
        app.server.post("/api/sessions").await
    })
    .await?;
    created.assert_status_ok();
    let created: Value = created.json();
    let id = created["data"]["session_id"].as_str().unwrap();
    let repository = app.repository();
    repository.update_title(id, "流式刷新标题").await?;
    repository
        .update_latest_message(
            id,
            "工具运行中",
            Utc.with_ymd_and_hms(2026, 10, 1, 12, 0, 0).unwrap(),
        )
        .await?;
    repository.update_status(id, SessionStatus::Running).await?;
    repository.update_unread_message_count(id, 4).await?;
    let details = app.server.get(&format!("/api/sessions/{id}")).await;
    details.assert_status_ok();

    assert!(
        timeout_at(
            first_received + Duration::from_secs(4),
            next_sessions_frame(&mut response, &mut buffered),
        )
        .await
        .is_err(),
        "连续快照之间应遵循五秒睡眠间隔"
    );
    let second = timeout_at(
        first_received + Duration::from_secs(7),
        next_sessions_frame(&mut response, &mut buffered),
    )
    .await??;
    let expected = json!({"sessions": [{
        "session_id": id,
        "title": "流式刷新标题",
        "latest_message": "工具运行中",
        "latest_message_at": "2026-10-01T12:00:00Z",
        "status": "running",
        "unread_message_count": 4
    }]});
    assert_eq!(second, expected);

    // 数据没有变化时也继续发送全量快照，保证持续订阅语义。
    let third = timeout(
        Duration::from_secs(7),
        next_sessions_frame(&mut response, &mut buffered),
    )
    .await??;
    assert_eq!(third, expected);
    assert_eq!(
        repository
            .get_by_id(id)
            .await?
            .unwrap()
            .unread_message_count,
        4
    );

    // 主动断开客户端后，仍可通过普通接口操作同一个会话。
    drop(response);
    timeout(Duration::from_secs(2), async {
        app.server.post(&format!("/api/sessions/{id}/delete")).await
    })
    .await?
    .assert_status_ok();
    Ok(())
}

#[tokio::test]
#[serial]
async fn clears_only_the_selected_unread_count_and_can_repeat() -> Result<()> {
    let app = TestApp::new().await?;
    let repository = app.repository();
    let session = Session {
        title: "保留标题与事件".into(),
        latest_message: "保留消息".into(),
        unread_message_count: 3,
        events: vec![Event::Title(TitleEvent::default())],
        status: SessionStatus::Running,
        ..Session::default()
    };
    let other = Session {
        unread_message_count: 5,
        ..Session::default()
    };
    repository.save(session.clone()).await?;
    repository.save(other.clone()).await?;

    for _ in 0..2 {
        let response = app
            .server
            .post(&format!(
                "/api/sessions/{}/clear-unread-message-count",
                session.id
            ))
            .await;
        response.assert_status_ok();
        assert_eq!(
            response.json::<Value>(),
            json!({"code": 200, "msg": "清除未读消息数成功", "data": null})
        );
        let mut expected = session.clone();
        expected.unread_message_count = 0;
        let actual = repository.get_by_id(&session.id).await?.unwrap();
        // 更新时间由数据库刷新，其余完整业务数据保持原值。
        expected.updated_at = actual.updated_at;
        // 数据库时间精度为微秒，使用实际写入后的创建时间。
        expected.created_at = actual.created_at;
        assert_eq!(actual, expected);
        assert_eq!(
            repository
                .get_by_id(&other.id)
                .await?
                .unwrap()
                .unread_message_count,
            5
        );
    }
    Ok(())
}

#[tokio::test]
#[serial]
async fn deletes_the_selected_session_and_reports_missing_on_repeat() -> Result<()> {
    let app = TestApp::new().await?;
    let repository = app.repository();
    let session = Session::default();
    let other = Session::default();
    repository.save(session.clone()).await?;
    repository.save(other.clone()).await?;
    let url = format!("/api/sessions/{}/delete", session.id);

    let deleted = app.server.post(&url).await;
    deleted.assert_status_ok();
    assert_eq!(
        deleted.json::<Value>(),
        json!({"code": 200, "msg": "删除任务会话成功", "data": null})
    );
    assert!(repository.get_by_id(&session.id).await?.is_none());
    let remaining = app.server.get("/api/sessions").await;
    remaining.assert_status_ok();
    let remaining: Value = remaining.json();
    assert_eq!(remaining["data"]["sessions"].as_array().unwrap().len(), 1);
    assert_eq!(remaining["data"]["sessions"][0]["session_id"], other.id);

    let repeated = app.server.post(&url).await;
    repeated.assert_status_not_found();
    assert_eq!(repeated.json::<Value>()["error"], "session.not_found");
    assert!(repeated.json::<Value>()["description"]
        .as_str()
        .unwrap()
        .contains(&session.id));
    Ok(())
}

#[tokio::test]
#[serial]
async fn gets_all_session_files_in_saved_order_without_mutating_the_session() -> Result<()> {
    let app = TestApp::new().await?;
    let repository = app.repository();
    // 文件列表直接读取会话保存的 Human/AI 附件，并保留同一路径的多个版本。
    let files = vec![
        File {
            filename: "用户资料.txt".into(),
            filepath: "/home/ubuntu/upload/用户资料.txt".into(),
            key: "uploads/human-source.txt".into(),
            extension: ".txt".into(),
            mime_type: "text/plain".into(),
            size: 128,
            ..File::default()
        },
        File {
            filename: "随机数.py".into(),
            filepath: "/home/ubuntu/随机数.py".into(),
            key: "outputs/first-version.py".into(),
            extension: ".py".into(),
            mime_type: "text/x-python".into(),
            size: 64,
            ..File::default()
        },
        File {
            filename: "随机数.py".into(),
            filepath: "/home/ubuntu/随机数.py".into(),
            key: "outputs/second-version.py".into(),
            extension: ".py".into(),
            mime_type: "text/x-python".into(),
            size: 256,
            ..File::default()
        },
    ];
    let session = Session {
        files: files.clone(),
        events: vec![Event::Message(MessageEvent::default())],
        unread_message_count: 7,
        status: SessionStatus::Running,
        task_id: Some("保留任务".into()),
        sandbox_id: Some("保留沙箱".into()),
        ..Session::default()
    };
    repository.save(session.clone()).await?;
    let before = repository.get_by_id(&session.id).await?.unwrap();
    // Null cache 和缺失模型配置证明该接口独立于 Agent、Redis 和对象存储下载。
    let response = app
        .server
        .get(&format!("/api/sessions/{}/files", session.id))
        .await;
    response.assert_status_ok();
    assert_eq!(
        response.json::<Value>(),
        json!({
            "code": 200, "msg": "获取会话文件列表成功", "data": {"files": files}
        })
    );
    for file in response.json::<Value>()["data"]["files"]
        .as_array()
        .unwrap()
    {
        assert_eq!(file.as_object().unwrap().len(), 7);
    }
    assert_eq!(repository.get_by_id(&session.id).await?.unwrap(), before);

    let blank = Session::default();
    repository.save(blank.clone()).await?;
    let response = app
        .server
        .get(&format!("/api/sessions/{}/files", blank.id))
        .await;
    response.assert_status_ok();
    assert_eq!(
        response.json::<Value>(),
        json!({
            "code": 200, "msg": "获取会话文件列表成功", "data": {"files": []}
        })
    );
    let missing = app
        .server
        .get(&format!("/api/sessions/{}/files", uuid::Uuid::new_v4()))
        .await;
    missing.assert_status_internal_server_error();
    assert_eq!(missing.json::<Value>()["error"], "internal_server_error");
    Ok(())
}

#[tokio::test]
#[serial]
async fn sandbox_content_routes_validate_requests_and_preserve_sessions_without_sandboxes(
) -> Result<()> {
    let app = TestApp::new().await?;
    let repository = app.repository();

    // 任务会话路径使用 UUID；Shell 会话标识是沙箱内部的任意字符串。
    for sandbox_id in [None, Some(String::new())] {
        let session = Session {
            sandbox_id,
            status: SessionStatus::Running,
            unread_message_count: 3,
            events: vec![Event::Title(TitleEvent::default())],
            files: vec![File::default()],
            ..Session::default()
        };
        repository.save(session.clone()).await?;
        let before = repository.get_by_id(&session.id).await?.unwrap();
        for (action, body) in [
            ("file", json!({"filepath": "/home/ubuntu/结果.txt"})),
            ("shell", json!({"session_id": "manus-shell"})),
            ("shell", json!({"session_id": ""})),
        ] {
            // 缺少沙箱时直接返回404，避免访问 Docker 或固定端口服务。
            let response = app
                .server
                .post(&format!("/api/sessions/{}/{action}", session.id))
                .json(&body)
                .await;
            response.assert_status_not_found();
            assert_eq!(
                response.json::<Value>()["error"],
                "session.sandbox_not_found"
            );
            assert_eq!(
                response.json::<Value>()["description"],
                "当前会话无沙箱环境"
            );
            assert_eq!(repository.get_by_id(&session.id).await?.unwrap(), before);
        }
    }

    let missing = uuid::Uuid::new_v4();
    for (action, field, valid_body) in [
        (
            "file",
            "filepath",
            json!({"filepath": "/home/ubuntu/result.txt"}),
        ),
        ("shell", "session_id", json!({"session_id": "manus-shell"})),
    ] {
        let response = app
            .server
            .post(&format!("/api/sessions/invalid/{action}"))
            .json(&valid_body)
            .await;
        response.assert_status_bad_request();
        assert_eq!(response.json::<Value>()["error"], "session.invalid_id");

        let url = format!("/api/sessions/{missing}/{action}");
        let response = app.server.post(&url).json(&valid_body).await;
        response.assert_status_internal_server_error();
        assert_eq!(response.json::<Value>()["error"], "internal_server_error");

        // Json 提取器在进入服务前拒绝缺字段、null 和错误类型。
        for body in [
            json!({}),
            json!({(field): null}),
            json!({(field): 1}),
            json!({(field): []}),
        ] {
            let response = app.server.post(&url).json(&body).await;
            assert_eq!(response.status_code(), 422);
        }
    }
    Ok(())
}

#[tokio::test]
#[serial]
async fn stops_sessions_with_no_registered_task_and_preserves_their_data_on_repeat() -> Result<()> {
    let app = TestApp::with_agent_config(true).await?;
    let repository = app.repository();
    for (status, task_id) in [
        (SessionStatus::Pending, None),
        (
            SessionStatus::Running,
            Some(uuid::Uuid::new_v4().to_string()),
        ),
        (
            SessionStatus::Waiting,
            Some(uuid::Uuid::new_v4().to_string()),
        ),
        (
            SessionStatus::Completed,
            Some(uuid::Uuid::new_v4().to_string()),
        ),
    ] {
        let session = Session {
            status,
            task_id,
            sandbox_id: Some("保留沙箱".into()),
            title: "保留会话".into(),
            unread_message_count: 3,
            events: vec![Event::Title(TitleEvent::default())],
            files: vec![File::default()],
            ..Session::default()
        };
        repository.save(session.clone()).await?;
        let before = repository.get_by_id(&session.id).await?.unwrap();
        for _ in 0..2 {
            // 无请求体；已释放的 Task 不触发任务重建，也不需要 Redis 网络连接。
            let response = app
                .server
                .post(&format!("/api/sessions/{}/stop", session.id))
                .await;
            response.assert_status_ok();
            assert_eq!(
                response.json::<Value>(),
                json!({
                    "code": 200, "msg": "停止任务会话成功", "data": null
                })
            );
            let actual = repository.get_by_id(&session.id).await?.unwrap();
            let mut expected = before.clone();
            expected.status = SessionStatus::Completed;
            expected.updated_at = actual.updated_at;
            assert_eq!(actual, expected);
        }
    }
    let missing = app
        .server
        .post(&format!("/api/sessions/{}/stop", uuid::Uuid::new_v4()))
        .await;
    missing.assert_status_internal_server_error();
    assert_eq!(missing.json::<Value>()["error"], "internal_server_error");
    Ok(())
}

#[tokio::test]
#[serial]
async fn rejects_invalid_ids_and_keeps_database_failures_distinct_from_missing_sessions(
) -> Result<()> {
    let app = TestApp::with_agent_config(true).await?;
    let invalid = app.server.get("/api/sessions/invalid").await;
    invalid.assert_status_bad_request();
    assert_eq!(invalid.json::<Value>()["error"], "session.invalid_id");
    let invalid = app.server.get("/api/sessions/invalid/files").await;
    invalid.assert_status_bad_request();
    assert_eq!(invalid.json::<Value>()["error"], "session.invalid_id");
    for action in ["delete", "clear-unread-message-count", "stop"] {
        let invalid = app
            .server
            .post(&format!("/api/sessions/invalid/{action}"))
            .await;
        invalid.assert_status_bad_request();
        assert_eq!(invalid.json::<Value>()["error"], "session.invalid_id");
    }
    let missing = uuid::Uuid::new_v4();
    app.server
        .post(&format!("/api/sessions/{missing}/delete"))
        .await
        .assert_status_not_found();

    // 仅破坏本测试创建的临时表，证明数据库故障不会伪装成空列表或404。
    app.database
        .db
        .execute_unprepared("DROP TABLE sessions")
        .await?;
    for response in [
        app.server.post("/api/sessions").await,
        app.server.get("/api/sessions").await,
        app.server.get(&format!("/api/sessions/{missing}")).await,
        app.server
            .get(&format!("/api/sessions/{missing}/files"))
            .await,
        app.server
            .post(&format!("/api/sessions/{missing}/stop"))
            .await,
        app.server
            .post(&format!("/api/sessions/{missing}/file"))
            .json(&json!({"filepath": "/home/ubuntu/result.txt"}))
            .await,
        app.server
            .post(&format!("/api/sessions/{missing}/shell"))
            .json(&json!({"session_id": "manus-shell"}))
            .await,
        app.server
            .post(&format!("/api/sessions/{missing}/delete"))
            .await,
        app.server
            .post(&format!(
                "/api/sessions/{missing}/clear-unread-message-count"
            ))
            .await,
    ] {
        response.assert_status_internal_server_error();
        assert_eq!(response.json::<Value>()["error"], "internal_server_error");
        assert!(!response.text().contains("SELECT") && !response.text().contains("relation"));
    }
    Ok(())
}

#[test]
#[serial]
fn registers_management_operations_and_their_response_schemas() -> Result<()> {
    server::openapi::clear_routes();
    server::controllers::sessions::routes();
    let document = serde_json::to_value(server::openapi::document())?;
    let paths = &document["paths"];
    assert_eq!(paths.as_object().unwrap().len(), 10);
    for (path, method) in [
        ("/api/sessions", "post"),
        ("/api/sessions", "get"),
        ("/api/sessions/{session_id}", "get"),
        (
            "/api/sessions/{session_id}/clear-unread-message-count",
            "post",
        ),
        ("/api/sessions/{session_id}/delete", "post"),
        ("/api/sessions/{session_id}/stop", "post"),
        ("/api/sessions/{session_id}/files", "get"),
        ("/api/sessions/{session_id}/file", "post"),
        ("/api/sessions/{session_id}/shell", "post"),
    ] {
        let operation = &paths[path][method];
        assert_eq!(operation["tags"], json!(["会话模块"]));
        let reference = operation["responses"]["200"]["content"]["application/json"]["schema"]
            ["$ref"]
            .as_str()
            .unwrap();
        let schema = document.pointer(&reference[1..]).unwrap();
        assert!(schema["properties"].get("data").is_some());
    }
    assert!(
        paths["/api/sessions/{session_id}/delete"]["post"]["responses"]
            .get("404")
            .is_some()
    );
    let item = &document["components"]["schemas"]["ListSessionItem"]["properties"];
    assert_eq!(item.as_object().unwrap().len(), 6);
    assert!(item.get("session_id").is_some());
    let detail = &paths["/api/sessions/{session_id}"]["get"];
    assert!(detail["responses"].get("404").is_some());
    let schema = &document["components"]["schemas"]["GetSessionResponse"]["properties"];
    assert_eq!(schema.as_object().unwrap().len(), 4);
    assert!(schema.get("events").is_some());
    let stream = &paths["/api/sessions/stream"]["post"];
    assert_eq!(stream["tags"], json!(["会话模块"]));
    assert!(stream["responses"]["200"]["content"]
        .get("text/event-stream")
        .is_some());
    assert!(stream.get("requestBody").is_none());
    let stop = &paths["/api/sessions/{session_id}/stop"]["post"];
    assert!(stop.get("requestBody").is_none());
    assert!(stop["responses"].get("500").is_some());
    assert_eq!(
        stop["responses"]["200"]["content"]["application/json"]["schema"],
        paths["/api/sessions/{session_id}/delete"]["post"]["responses"]["200"]["content"]
            ["application/json"]["schema"]
    );
    let files = &document["components"]["schemas"]["GetSessionFilesResponse"]["properties"];
    assert_eq!(files.as_object().unwrap().len(), 1);
    assert_eq!(files["files"]["type"], "array");
    assert_eq!(
        files["files"]["items"]["$ref"],
        "#/components/schemas/FileInfoResponse"
    );
    assert_eq!(
        document["components"]["schemas"]["FileInfoResponse"]["properties"]
            .as_object()
            .unwrap()
            .len(),
        7
    );
    // 详情嵌套事件、计划、附件等响应结构，文档中的每个本地引用均应有定义。
    assert_schema_references_resolve(&document, &document);
    Ok(())
}

#[test]
#[serial]
fn documents_sandbox_read_requests_and_typed_content_responses() -> Result<()> {
    server::openapi::clear_routes();
    server::controllers::sessions::routes();
    let document = serde_json::to_value(server::openapi::document())?;
    // Utoipa 在泛型响应包裹中内联 data，按最终文档验证实际载荷结构。
    let response_schema = |action: &str| {
        let path = format!("/api/sessions/{{session_id}}/{action}");
        let reference = document["paths"][&path]["post"]["responses"]["200"]["content"]
            ["application/json"]["schema"]["$ref"]
            .as_str()
            .unwrap();
        &document.pointer(&reference[1..]).unwrap()["properties"]["data"]
    };
    for (action, request_name, request_field, response_fields) in [
        (
            "file",
            "FileReadRequest",
            "filepath",
            vec!["filepath", "content"],
        ),
        (
            "shell",
            "ShellReadRequest",
            "session_id",
            vec!["session_id", "output", "console_records"],
        ),
    ] {
        let path = format!("/api/sessions/{{session_id}}/{action}");
        let operation = &document["paths"][&path]["post"];
        assert_eq!(operation["tags"], json!(["会话模块"]));
        assert_eq!(operation["requestBody"]["required"], true);
        assert_eq!(
            operation["requestBody"]["content"]["application/json"]["schema"]["$ref"],
            format!("#/components/schemas/{request_name}")
        );
        let request = &document["components"]["schemas"][request_name];
        assert_eq!(request["properties"].as_object().unwrap().len(), 1);
        assert_eq!(request["required"], json!([request_field]));
        assert_eq!(request["properties"][request_field]["type"], "string");
        // Body 中的 Shell session_id 保持普通字符串；路径参数明确表示任务会话。
        assert!(request["properties"][request_field].get("format").is_none());
        assert!(operation["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .any(|parameter| parameter["name"] == "session_id"
                && parameter["in"] == "path"
                && parameter["required"] == true));
        for status in ["400", "404", "422", "500"] {
            assert!(operation["responses"].get(status).is_some());
        }

        let response = response_schema(action);
        assert_eq!(
            response["properties"].as_object().unwrap().len(),
            response_fields.len()
        );
        for field in response_fields {
            assert!(response["properties"].get(field).is_some());
        }
    }

    let file = response_schema("file");
    assert_eq!(file["required"], json!(["filepath", "content"]));
    let shell = response_schema("shell");
    assert_eq!(shell["required"], json!(["session_id", "output"]));
    // console_records 可省略，反序列化时使用空列表，记录中的三个文本字段均必填。
    assert_eq!(shell["properties"]["console_records"]["type"], "array");
    let record_reference = shell["properties"]["console_records"]["items"]["$ref"]
        .as_str()
        .unwrap();
    let record = document.pointer(&record_reference[1..]).unwrap();
    assert_eq!(record["required"], json!(["ps1", "command", "output"]));
    assert_eq!(record["properties"].as_object().unwrap().len(), 3);
    for field in ["ps1", "command", "output"] {
        assert_eq!(record["properties"][field]["type"], "string");
    }
    assert_schema_references_resolve(&document, &document);
    Ok(())
}

fn assert_schema_references_resolve(value: &Value, document: &Value) {
    match value {
        Value::Object(fields) => {
            if let Some(pointer) = fields
                .get("$ref")
                .and_then(Value::as_str)
                .and_then(|reference| reference.strip_prefix('#'))
            {
                assert!(
                    document.pointer(pointer).is_some(),
                    "OpenAPI 缺少引用的结构：#{pointer}"
                );
            }
            for child in fields.values() {
                assert_schema_references_resolve(child, document);
            }
        }
        Value::Array(items) => {
            for item in items {
                assert_schema_references_resolve(item, document);
            }
        }
        _ => {}
    }
}

#[tokio::test]
#[serial]
async fn chat_without_message_subscribes_and_missing_session_returns_uniform_error() -> Result<()> {
    let app = TestApp::with_agent_config(true).await?;
    let repository = app.repository();
    let session = Session::default();
    repository.save(session.clone()).await?;
    let before = repository.get_by_id(&session.id).await?.unwrap();
    let url = format!("/api/sessions/{}/chat", session.id);

    for body in [
        json!({}),
        json!({"message": "", "attachments": ["file-1"], "event_id": "123-0", "timestamp": 123}),
    ] {
        let response = app.server.post(&url).json(&body).await;
        response.assert_status_ok();
        assert_eq!(response.header("content-type"), "text/event-stream");
        // 空白会话没有任务，订阅结束时保持空事件流并清除未读。
        assert!(response.text().is_empty());
        let actual = repository.get_by_id(&session.id).await?.unwrap();
        let mut expected = before.clone();
        expected.updated_at = actual.updated_at;
        assert_eq!(actual, expected);
    }

    repository
        .update_status(&session.id, SessionStatus::Running)
        .await?;
    let response = app
        .server
        .post(&format!("/api/sessions/{}/chat", uuid::Uuid::new_v4()))
        .json(&json!({"message": "会话已不存在"}))
        .await;
    response.assert_status_ok();
    let body = response.text();
    assert!(body.contains("event: error"));
    let data = body
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap();
    let event: Value = serde_json::from_str(data)?;
    assert_eq!(event.as_object().unwrap().len(), 3);
    assert!(event["event_id"].is_string() && event["created_at"].is_i64());
    assert!(event["error"].as_str().unwrap().contains("任务会话不存在"));
    let after = repository.get_by_id(&session.id).await?.unwrap();
    assert_eq!(after.status, SessionStatus::Running);
    assert!(after.task_id.is_none() && after.sandbox_id.is_none());
    Ok(())
}

#[tokio::test]
#[serial]
async fn chat_rejects_bad_requests_and_reports_initialization_failures() -> Result<()> {
    let app = TestApp::new().await?;
    app.server
        .post("/api/sessions/invalid/chat")
        .json(&json!({}))
        .await
        .assert_status_bad_request();
    let url = format!("/api/sessions/{}/chat", uuid::Uuid::new_v4());
    let invalid_time = app
        .server
        .post(&url)
        .json(&json!({"timestamp": i64::MAX}))
        .await;
    invalid_time.assert_status_bad_request();
    assert_eq!(
        invalid_time.json::<Value>()["error"],
        "session.invalid_timestamp"
    );
    let invalid = app
        .server
        .post(&url)
        .json(&json!({"attachments": "wrong-type"}))
        .await;
    assert_eq!(invalid.status_code(), 422);
    // 本夹具使用 Null cache，缺少 Agent 所需 Redis 配置时走已有安全错误出口。
    let response = app.server.post(&url).json(&json!({})).await;
    response.assert_status_internal_server_error();
    assert_eq!(response.json::<Value>()["error"], "internal_server_error");
    Ok(())
}

#[test]
#[serial]
fn documents_chat_json_request_and_event_stream_response() -> Result<()> {
    server::openapi::clear_routes();
    server::controllers::sessions::routes();
    let document = serde_json::to_value(server::openapi::document())?;
    let chat = &document["paths"]["/api/sessions/{session_id}/chat"]["post"];
    assert_eq!(chat["tags"], json!(["会话模块"]));
    assert!(chat["responses"]["200"]["content"]
        .get("text/event-stream")
        .is_some());
    assert!(chat["requestBody"]["content"]
        .get("application/json")
        .is_some());
    let schema = &document["components"]["schemas"]["ChatRequest"];
    assert_eq!(schema["properties"].as_object().unwrap().len(), 4);
    assert!(schema
        .get("required")
        .is_none_or(|required| required.as_array().unwrap().is_empty()));
    Ok(())
}
