//! 文件服务组合真实仓库和存储适配器，并保持可区分的缺失、数据库与存储错误。

#[path = "support/file_database.rs"]
mod file_database;

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use loco_rs::storage::{drivers, Storage, StorageError};
use server::{
    application::services::FileService,
    domain::{
        external::{FileNotFound, InvalidFileUpload, UploadFile},
        models::File,
        repositories::FileRepository,
    },
    infrastructure::{external::LocoFileStorage, repositories::SeaOrmFileRepository},
};

#[tokio::test]
async fn gets_uploaded_metadata_and_downloads_content_through_file_service() -> Result<()> {
    let fixture = file_database::TestDatabase::new().await?;
    let repository: Arc<dyn FileRepository> =
        Arc::new(SeaOrmFileRepository::new(fixture.db.clone()));
    let storage = Arc::new(LocoFileStorage::new(
        Arc::new(Storage::single(drivers::mem::new())),
        repository.clone(),
    ));
    let service = FileService::new(storage, repository);

    let file = service
        .upload_file(UploadFile {
            filename: "学习笔记.md".to_string(),
            mime_type: Some("text/markdown".to_string()),
            content: Bytes::from_static(b"# notes\n"),
        })
        .await?;
    assert_eq!(service.get_file_info(&file.id).await?, file);
    let (stream, downloaded) = service.download_file(&file.id).await?;
    assert_eq!(downloaded, file);
    let chunks = stream.try_collect::<Vec<_>>().await?;
    assert_eq!(chunks.concat(), b"# notes\n");

    let missing_id = uuid::Uuid::new_v4().to_string();
    let info_error = service.get_file_info(&missing_id).await.unwrap_err();
    let download_error = service.download_file(&missing_id).await.err().unwrap();
    assert_eq!(
        info_error.downcast_ref::<FileNotFound>().unwrap().file_id,
        missing_id
    );
    assert!(download_error.is::<FileNotFound>());
    Ok(())
}

#[derive(Debug, thiserror::Error)]
#[error("injected repository read failure")]
struct RepositoryReadFailure;

struct FailingReadRepository;

#[async_trait]
impl FileRepository for FailingReadRepository {
    async fn save(&self, _file: File) -> Result<()> {
        panic!("此测试不应保存文件记录")
    }

    async fn get_by_id(&self, _file_id: &str) -> Result<Option<File>> {
        Err(RepositoryReadFailure.into())
    }
}

#[tokio::test]
async fn does_not_report_repository_failure_as_file_not_found() {
    let repository: Arc<dyn FileRepository> = Arc::new(FailingReadRepository);
    let storage = Arc::new(LocoFileStorage::new(
        Arc::new(Storage::single(drivers::mem::new())),
        repository.clone(),
    ));
    let service = FileService::new(storage, repository);
    let id = uuid::Uuid::new_v4().to_string();
    for error in [
        service.get_file_info(&id).await.unwrap_err(),
        service.download_file(&id).await.err().unwrap(),
    ] {
        assert!(error.is::<RepositoryReadFailure>());
        assert!(!error.is::<FileNotFound>());
    }
}

#[tokio::test]
async fn preserves_storage_failure_and_invalid_upload_types() {
    let repository: Arc<dyn FileRepository> = Arc::new(FailingReadRepository);
    let storage = Arc::new(LocoFileStorage::new(
        Arc::new(Storage::single(drivers::null::new())),
        repository.clone(),
    ));
    let service = FileService::new(storage, repository);
    let error = service
        .upload_file(UploadFile {
            filename: "report.txt".to_string(),
            mime_type: None,
            content: Bytes::from_static(b"data"),
        })
        .await
        .unwrap_err();
    assert!(error.is::<StorageError>());
    assert!(!error.is::<FileNotFound>());
    assert!(!error.is::<InvalidFileUpload>());

    let error = service
        .upload_file(UploadFile {
            filename: String::new(),
            mime_type: None,
            content: Bytes::new(),
        })
        .await
        .unwrap_err();
    assert!(error.is::<InvalidFileUpload>());
}
