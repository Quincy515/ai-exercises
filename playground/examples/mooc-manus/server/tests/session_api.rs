//! 会话 API 集成测试：复用私有临时 PostgreSQL，绕过应用 boot 和现有数据库。

#[path = "support/file_database.rs"]
mod file_database;

use anyhow::{ensure, Result};
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
        models::{Event, Session, SessionStatus, TitleEvent},
        repositories::SessionRepository,
    },
    infrastructure::repositories::SeaOrmSessionRepository,
};

struct TestApp {
    server: TestServer,
    database: file_database::TestDatabase,
}

impl TestApp {
    async fn new() -> Result<Self> {
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

        let config: Config = serde_json::from_value(json!({
            "logger": { "enable": false, "level": "info", "format": "compact" },
            "server": { "port": 0, "host": "http://localhost" },
            "database": {
                "uri": "unused", "enable_logging": false,
                "min_connections": 1, "max_connections": 1,
                "connect_timeout": 10, "idle_timeout": 10
            }
        }))?;
        let ctx = AppContext::builder(Environment::Test, database.db.clone(), config).build();
        // 通过真实应用路由覆盖依赖注入、控制器、服务及仓库的完整链路。
        let router = App::routes(&ctx).to_router::<App>(ctx, Router::new())?;
        Ok(Self {
            server: TestServer::new(router)?,
            database,
        })
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
async fn rejects_invalid_ids_and_keeps_database_failures_distinct_from_missing_sessions(
) -> Result<()> {
    let app = TestApp::new().await?;
    for action in ["delete", "clear-unread-message-count"] {
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
fn registers_all_four_operations_and_their_response_schemas() -> Result<()> {
    server::openapi::clear_routes();
    server::controllers::sessions::routes();
    let document = serde_json::to_value(server::openapi::document())?;
    let paths = &document["paths"];
    assert_eq!(paths.as_object().unwrap().len(), 3);
    for (path, method) in [
        ("/api/sessions", "post"),
        ("/api/sessions", "get"),
        (
            "/api/sessions/{session_id}/clear-unread-message-count",
            "post",
        ),
        ("/api/sessions/{session_id}/delete", "post"),
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
    Ok(())
}
