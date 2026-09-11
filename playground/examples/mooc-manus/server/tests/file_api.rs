//! 文件接口的 HTTP 集成测试：临时 PostgreSQL + 内存存储，不启动或清空应用数据库。

#[path = "support/file_database.rs"]
mod file_database;

use std::sync::Arc;

use anyhow::Result;
use axum::{
    http::{header, StatusCode},
    Router,
};
use axum_test::{
    multipart::{MultipartForm, Part},
    TestServer,
};
use bytes::Bytes;
use loco_rs::{
    app::{AppContext, Hooks, SharedStore},
    cache::{self, Cache},
    config::Config,
    environment::Environment,
    storage::{drivers, Storage},
};
use sea_orm::{ConnectionTrait, EntityTrait, PaginatorTrait};
use serde_json::{json, Value};
use serial_test::serial;
use server::{app::App, controllers::files::MAX_UPLOAD_BODY_SIZE, models::files::Entity};

struct TestApp {
    server: TestServer,
    database: file_database::TestDatabase,
    storage: Arc<Storage>,
}

impl TestApp {
    async fn new(storage: Storage) -> Result<Self> {
        let database = file_database::TestDatabase::new().await?;
        // 显式构造配置，不读取 DATABASE_URL，也不调用带迁移/清表行为的应用 boot。
        let config: Config = serde_json::from_value(json!({
            "logger": { "enable": false, "level": "info", "format": "compact" },
            "server": { "port": 0, "host": "http://localhost" },
            "database": {
                "uri": "unused", "enable_logging": false,
                "min_connections": 1, "max_connections": 1,
                "connect_timeout": 10, "idle_timeout": 10
            }
        }))?;
        let storage = Arc::new(storage);
        let ctx = AppContext {
            environment: Environment::Test,
            db: database.db.clone(),
            config,
            queue_provider: None,
            mailer: None,
            storage: Arc::clone(&storage),
            cache: Arc::new(Cache::new(cache::drivers::null::new())),
            shared_store: Arc::new(SharedStore::default()),
        };
        // 使用真实应用路由及 Loco 默认中间件，覆盖路由注册和上传限制的组合行为。
        let router = App::routes(&ctx).to_router::<App>(ctx, Router::new())?;
        let server = TestServer::new(router)?;
        Ok(Self {
            server,
            database,
            storage,
        })
    }

    async fn count_files(&self) -> Result<u64> {
        Ok(Entity::find().count(&self.database.db).await?)
    }
}

fn form(filename: &str, content: impl Into<Bytes>) -> MultipartForm {
    MultipartForm::new().add_part("file", Part::bytes(content).file_name(filename))
}

#[tokio::test]
#[serial]
async fn reports_missing_objects_before_starting_the_download_response() -> Result<()> {
    let app = TestApp::new(Storage::single(drivers::mem::new())).await?;
    let uploaded = app
        .server
        .post("/api/files")
        .multipart(form("lost.txt", "content"))
        .await;
    uploaded.assert_status_ok();
    let uploaded: Value = uploaded.json();
    let key = uploaded["data"]["key"].as_str().unwrap();
    let id = uploaded["data"]["id"].as_str().unwrap();
    app.storage.delete(std::path::Path::new(key)).await?;

    // 数据库记录仍在；信息查询成功，但下载应返回安全的 500，而不是 200 后断流。
    app.server
        .get(&format!("/api/files/{id}"))
        .await
        .assert_status_ok();
    let response = app.server.get(&format!("/api/files/{id}/download")).await;
    response.assert_status_internal_server_error();
    assert_eq!(response.json::<Value>()["error"], "internal_server_error");
    assert!(!response.text().contains(key));
    Ok(())
}

#[tokio::test]
#[serial]
async fn uploads_queries_and_downloads_binary_content_with_chinese_filename() -> Result<()> {
    let app = TestApp::new(Storage::single(drivers::mem::new())).await?;
    let content = Bytes::from_static(b"%PDF\0\xff\x80\r\n");
    let multipart = MultipartForm::new()
        .add_text("description", "普通表单字段")
        .add_part(
            "file",
            Part::bytes(content.clone())
                .file_name("学习 资料#1.pdf")
                .mime_type("application/pdf"),
        );
    let response = app.server.post("/api/files").multipart(multipart).await;
    response.assert_status_ok();
    let uploaded: Value = response.json();
    assert_eq!(uploaded["code"], 200);
    assert_eq!(uploaded["msg"], "上传文件成功");
    let file = &uploaded["data"];
    let file_id = file["id"].as_str().unwrap();
    uuid::Uuid::parse_str(file_id)?;
    assert_eq!(file["filename"], "学习 资料#1.pdf");
    assert_eq!(file["extension"], ".pdf");
    assert_eq!(file["mime_type"], "application/pdf");
    assert_eq!(file["filepath"], "");
    assert_eq!(file["size"], content.len());
    assert!(
        regex::Regex::new(&format!(r"^\d{{4}}/\d{{2}}/\d{{2}}/{file_id}\.pdf$"))?
            .is_match(file["key"].as_str().unwrap())
    );

    let info = app.server.get(&format!("/api/files/{file_id}")).await;
    info.assert_status_ok();
    let info: Value = info.json();
    assert_eq!(info["msg"], "获取文件信息成功");
    assert_eq!(&info["data"], file);

    let download = app
        .server
        .get(&format!("/api/files/{file_id}/download"))
        .await;
    download.assert_status_ok();
    assert_eq!(download.as_bytes(), &content);
    assert_eq!(download.header(header::CONTENT_TYPE), "application/pdf");
    assert_eq!(download.header(header::CONTENT_LENGTH), "9");
    assert_eq!(
        download.header(header::CONTENT_DISPOSITION),
        "attachment; filename*=utf-8''%E5%AD%A6%E4%B9%A0%20%E8%B5%84%E6%96%99%231.pdf"
    );
    assert_eq!(app.count_files().await?, 1);
    Ok(())
}

#[tokio::test]
#[serial]
async fn handles_empty_extensionless_files_without_a_content_type() -> Result<()> {
    let app = TestApp::new(Storage::single(drivers::mem::new())).await?;
    let body = "--empty-file\r\nContent-Disposition: form-data; name=\"file\"; filename=\"README\"\r\n\r\n\r\n--empty-file--\r\n";
    let response = app
        .server
        .post("/api/files")
        .content_type("multipart/form-data; boundary=empty-file")
        .bytes(Bytes::from_static(body.as_bytes()))
        .await;
    response.assert_status_ok();
    let uploaded: Value = response.json();
    assert_eq!(uploaded["data"]["extension"], "");
    assert_eq!(uploaded["data"]["mime_type"], "");
    assert_eq!(uploaded["data"]["size"], 0);
    let id = uploaded["data"]["id"].as_str().unwrap();
    let download = app.server.get(&format!("/api/files/{id}/download")).await;
    download.assert_status_ok();
    assert_eq!(
        download.header(header::CONTENT_TYPE),
        "application/octet-stream"
    );
    assert_eq!(download.header(header::CONTENT_LENGTH), "0");
    assert!(download.as_bytes().is_empty());
    Ok(())
}

#[tokio::test]
#[serial]
async fn rejects_invalid_multipart_without_persisting_partial_uploads() -> Result<()> {
    let app = TestApp::new(Storage::single(drivers::mem::new())).await?;
    app.server
        .post("/api/files")
        .json(&json!({"file": "text"}))
        .await
        .assert_status_bad_request();
    for multipart in [
        MultipartForm::new().add_text("other", "value"),
        MultipartForm::new().add_text("file", "not a file"),
        form("first.txt", "one").add_part("file", Part::bytes("two").file_name("second.txt")),
        form("..", "bad name"),
    ] {
        app.server
            .post("/api/files")
            .multipart(multipart)
            .await
            .assert_status_bad_request();
    }
    // 第一个字段已经完整，但第二个字段损坏；整个表单解析完成前不能写入对象/元数据。
    let malformed = "--broken\r\nContent-Disposition: form-data; name=\"file\"; filename=\"first.txt\"\r\n\r\none\r\n--broken\r\ninvalid header";
    app.server
        .post("/api/files")
        .content_type("multipart/form-data; boundary=broken")
        .bytes(Bytes::from_static(malformed.as_bytes()))
        .await
        .assert_status_bad_request();
    assert_eq!(app.count_files().await?, 0);
    Ok(())
}

#[tokio::test]
#[serial]
async fn distinguishes_invalid_ids_missing_files_and_internal_failures() -> Result<()> {
    let app = TestApp::new(Storage::single(drivers::null::new())).await?;
    let missing = uuid::Uuid::new_v4();
    for suffix in ["", "/download"] {
        let invalid = app.server.get(&format!("/api/files/invalid{suffix}")).await;
        invalid.assert_status_bad_request();
        let missing = app
            .server
            .get(&format!("/api/files/{missing}{suffix}"))
            .await;
        missing.assert_status_not_found();
        assert_eq!(missing.json::<Value>()["error"], "file.not_found");
    }
    let upload = app
        .server
        .post("/api/files")
        .multipart(form("failed.txt", "data"))
        .await;
    upload.assert_status_internal_server_error();
    assert_eq!(upload.json::<Value>()["error"], "internal_server_error");
    assert!(!upload.text().contains("null storage"));
    assert_eq!(app.count_files().await?, 0);

    app.database
        .db
        .execute_unprepared("DROP TABLE files")
        .await?;
    let response = app.server.get(&format!("/api/files/{missing}")).await;
    response.assert_status_internal_server_error();
    assert!(!response.text().contains("SELECT"));
    Ok(())
}

#[tokio::test]
#[serial]
async fn accepts_lesson_sized_files_and_enforces_the_upload_limit() -> Result<()> {
    let app = TestApp::new(Storage::single(drivers::mem::new())).await?;
    // 课程演示约 14 MB 的 PDF；必须覆盖 Loco/Axum 默认约 2 MB 的限制。
    let content = Bytes::from(vec![0x80; 14 * 1024 * 1024 + 100_000]);
    let response = app
        .server
        .post("/api/files")
        .multipart(form("large.pdf", content.clone()))
        .await;
    response.assert_status_ok();
    let uploaded: Value = response.json();
    assert_eq!(uploaded["data"]["size"], content.len());
    let id = uploaded["data"]["id"].as_str().unwrap();
    let download = app.server.get(&format!("/api/files/{id}/download")).await;
    assert_eq!(download.as_bytes(), &content);

    // 文件加上 multipart 边界超过整个请求的 32 MiB 上限。
    let oversized = Bytes::from(vec![0; MAX_UPLOAD_BODY_SIZE]);
    let response = app
        .server
        .post("/api/files")
        .multipart(form("too-large.bin", oversized))
        .await;
    assert_eq!(response.status_code(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(app.count_files().await?, 1);
    Ok(())
}

#[tokio::test]
#[serial]
async fn collects_upload_and_download_openapi_schemas_from_registered_routes() -> Result<()> {
    let _app = TestApp::new(Storage::single(drivers::mem::new())).await?;
    let (_, document) = loco_openapi::openapi::get_merged_router().split_for_parts();
    let document = serde_json::to_value(document)?;
    assert!(
        document["paths"]["/api/files"]["post"]["requestBody"]["content"]
            .get("multipart/form-data")
            .is_some()
    );
    assert_eq!(
        document["components"]["schemas"]["FileUploadRequest"]["properties"]["file"]["format"],
        "binary"
    );
    assert!(
        document["paths"]["/api/files/{file_id}"]["get"]["responses"]
            .get("404")
            .is_some()
    );
    let download = &document["paths"]["/api/files/{file_id}/download"]["get"]["responses"]["200"];
    assert!(download["content"]
        .get("application/octet-stream")
        .is_some());
    assert!(download["headers"].get("Content-Disposition").is_some());
    assert!(download["headers"].get("Content-Length").is_some());
    assert!(
        document["components"]["schemas"]["FileResponse"]["properties"]
            .get("data")
            .is_some()
    );
    Ok(())
}
