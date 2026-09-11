//! 文件仓库的真实 PostgreSQL 集成测试。
//!
//! 运行：`cargo test --test file_repository`。
//! 只使用夹具创建的临时实例，不加载应用配置或连接现有数据库。

#[path = "support/file_database.rs"]
mod file_database;

use anyhow::{Context, Result};
use file_database::TestDatabase;
use futures::future::try_join_all;
use sea_orm::{ConnectionTrait, DatabaseBackend, EntityTrait, PaginatorTrait, Statement, Value};
use server::{
    domain::{models::File, repositories::FileRepository},
    infrastructure::repositories::SeaOrmFileRepository,
    models::files::{Entity, Model},
};
use tokio::sync::Barrier;
use uuid::Uuid;

fn file() -> File {
    File {
        filename: "用户's自学笔记.md".to_string(),
        filepath: "/workspace/用户's自学笔记.md".to_string(),
        key: "uploads/2026/09/11/notes.md".to_string(),
        extension: "md".to_string(),
        mime_type: "text/markdown".to_string(),
        size: 1024,
        ..File::default()
    }
}

async fn row(fixture: &TestDatabase, file_id: &str) -> Result<Model> {
    Entity::find_by_uuid(file_id)?
        .one(&fixture.db)
        .await?
        .context("测试期望文件记录存在")
}

async fn execute(fixture: &TestDatabase, sql: &str, values: Vec<Value>) -> Result<()> {
    fixture
        .db
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            sql,
            values,
        ))
        .await?;
    Ok(())
}

#[tokio::test]
async fn creates_and_updates_all_file_fields_without_overwriting_database_metadata() -> Result<()> {
    let fixture = TestDatabase::new().await?;
    let repository = SeaOrmFileRepository::new(fixture.db.clone());
    let mut file = file();
    repository.save(file.clone()).await?;
    assert_eq!(repository.get_by_id(&file.id).await?, Some(file.clone()));

    let user_id = Uuid::new_v4();
    execute(
        &fixture,
        "UPDATE files SET user_id = $1, remark = $2, is_deleted = true, \
         updated_at = '2000-01-01T00:00:00Z' WHERE uuid = $3",
        vec![
            user_id.into(),
            "数据库独有备注".into(),
            Uuid::parse_str(&file.id)?.into(),
        ],
    )
    .await?;
    let before = row(&fixture, &file.id).await?;

    file.filename = "结果.txt".to_string();
    file.filepath = "/workspace/结果.txt".to_string();
    file.key = "uploads/2026/09/11/result.txt".to_string();
    file.extension = "txt".to_string();
    file.mime_type = "text/plain".to_string();
    file.size = i32::MAX as usize;
    repository.save(file.clone()).await?;

    let after = row(&fixture, &file.id).await?;
    assert_eq!(after.id, before.id);
    assert_eq!(after.uuid, before.uuid);
    assert_eq!(after.created_at, before.created_at);
    assert!(after.updated_at > before.updated_at);
    assert_eq!(after.user_id, Some(user_id));
    assert_eq!(after.is_deleted, Some(true));
    assert_eq!(after.remark, before.remark);
    assert_eq!(repository.get_by_id(&file.id).await?, Some(file));
    assert_eq!(Entity::find().count(&fixture.db).await?, 1);
    Ok(())
}

#[tokio::test]
async fn concurrent_saves_of_one_uuid_create_only_one_complete_record() -> Result<()> {
    let fixture = TestDatabase::new().await?;
    let repository = SeaOrmFileRepository::new(fixture.db.clone());
    let file = file();
    let barrier = Barrier::new(12);
    try_join_all((0..12).map(|_| {
        let file = file.clone();
        let repository = &repository;
        let barrier = &barrier;
        async move {
            barrier.wait().await;
            repository.save(file).await
        }
    }))
    .await?;

    assert_eq!(Entity::find().count(&fixture.db).await?, 1);
    assert_eq!(repository.get_by_id(&file.id).await?, Some(file));
    Ok(())
}

#[tokio::test]
async fn distinguishes_missing_files_invalid_data_and_database_errors() -> Result<()> {
    let fixture = TestDatabase::new().await?;
    let repository = SeaOrmFileRepository::new(fixture.db.clone());
    assert!(repository
        .get_by_id(&Uuid::new_v4().to_string())
        .await?
        .is_none());
    assert!(repository.get_by_id("invalid-uuid").await.is_err());

    let file = file();
    repository.save(file.clone()).await?;
    let before = row(&fixture, &file.id).await?;
    assert!(repository
        .save(File {
            size: i32::MAX as usize + 1,
            ..file.clone()
        })
        .await
        .is_err());
    assert_eq!(row(&fixture, &file.id).await?, before);
    assert!(repository
        .save(File {
            id: "invalid-uuid".to_string(),
            ..file.clone()
        })
        .await
        .is_err());
    assert_eq!(Entity::find().count(&fixture.db).await?, 1);

    // 兼容历史可空列，但非法负数必须返回错误，不能包装成未找到。
    execute(
        &fixture,
        "UPDATE files SET filename = NULL, filepath = NULL, key = NULL, \
         extension = NULL, mime_type = NULL, size = NULL WHERE uuid = $1",
        vec![Uuid::parse_str(&file.id)?.into()],
    )
    .await?;
    assert_eq!(
        repository.get_by_id(&file.id).await?,
        Some(File {
            id: file.id.clone(),
            ..File::default()
        })
    );
    execute(
        &fixture,
        "UPDATE files SET size = -1 WHERE uuid = $1",
        vec![Uuid::parse_str(&file.id)?.into()],
    )
    .await?;
    assert!(repository.get_by_id(&file.id).await.is_err());

    // 只移除本测试临时实例中的表，验证底层数据库错误不会被吞成 None。
    fixture.db.execute_unprepared("DROP TABLE files").await?;
    assert!(repository.get_by_id(&file.id).await.is_err());
    Ok(())
}
