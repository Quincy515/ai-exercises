//! 文件上传、元数据持久化和流式下载；仅使用临时 PostgreSQL 与本地/内存驱动。

#[path = "support/file_database.rs"]
mod file_database;

use std::{path::Path, sync::Arc};

use anyhow::{bail, Result};
use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use loco_rs::storage::{drivers, Storage};
use server::{
    domain::{
        external::{FileNotFound, FileStorage, FileStream, InvalidFileUpload, UploadFile},
        models::File,
        repositories::FileRepository,
    },
    infrastructure::{
        external::LocoFileStorage, repositories::SeaOrmFileRepository, settings::StorageSettings,
    },
};
use tokio::sync::Mutex;

fn upload(filename: &str, content: &'static [u8]) -> UploadFile {
    UploadFile {
        filename: filename.to_string(),
        mime_type: None,
        content: Bytes::from_static(content),
    }
}

async fn read_all(stream: FileStream) -> Result<Vec<u8>> {
    Ok(stream
        .try_fold(Vec::new(), |mut data, chunk| async move {
            data.extend_from_slice(&chunk);
            Ok(data)
        })
        .await?)
}

#[tokio::test]
async fn uploads_and_downloads_binary_files_through_database_and_local_storage() -> Result<()> {
    let fixture = file_database::TestDatabase::new().await?;
    let directory = tempfile::tempdir()?;
    let storage = Arc::new(
        StorageSettings::Local {
            path: directory.path().to_owned(),
        }
        .build()?,
    );
    let repository = Arc::new(SeaOrmFileRepository::new(fixture.db.clone()));
    let service = LocoFileStorage::new(storage.clone(), repository.clone());

    let mut source = upload(r"C:\fakepath\学习资料.pdf", b"%PDF\0\xff\x80\r\n");
    source.mime_type = Some("application/pdf".to_string());
    let mut file = service.upload_file(source).await?;
    assert_eq!(file.filename, "学习资料.pdf");
    assert_eq!(file.extension, ".pdf");
    assert_eq!(file.mime_type, "application/pdf");
    assert_eq!(file.size, 9);
    assert_eq!(file.filepath, "");
    let (date_path, object_name) = file.key.rsplit_once('/').unwrap();
    assert!(chrono::NaiveDate::parse_from_str(date_path, "%Y/%m/%d").is_ok());
    assert_eq!(object_name, format!("{}.pdf", file.id));
    assert_eq!(repository.get_by_id(&file.id).await?, Some(file.clone()));
    assert_eq!(
        std::fs::read(directory.path().join(&file.key))?,
        b"%PDF\0\xff\x80\r\n"
    );

    // 同名附件独立保存；下载只使用 key，不使用后来写入的沙箱 filepath。
    let second = service
        .upload_file(upload("学习资料.pdf", b"second"))
        .await?;
    assert_ne!(file.id, second.id);
    assert_ne!(file.key, second.key);
    file.filepath = "/workspace/学习资料.pdf".to_string();
    repository.save(file.clone()).await?;
    let (stream, metadata) = service.download_file(&file.id).await?;
    assert_eq!(metadata, file);
    assert_eq!(read_all(stream).await?, b"%PDF\0\xff\x80\r\n");
    let (stream, _) = service.download_file(&second.id).await?;
    assert_eq!(read_all(stream).await?, b"second");

    let missing_error = service
        .download_file(&uuid::Uuid::new_v4().to_string())
        .await
        .err()
        .unwrap();
    assert!(missing_error.is::<FileNotFound>());
    assert!(service.download_file("invalid-uuid").await.is_err());
    // 对象缺失可能在打开流或消费流时暴露，两种错误都不能当成空文件。
    storage.delete(Path::new(&file.key)).await?;
    if let Ok((stream, _)) = service.download_file(&file.id).await {
        assert!(read_all(stream).await.is_err());
    }
    file.key.clear();
    repository.save(file.clone()).await?;
    assert!(service.download_file(&file.id).await.is_err());
    Ok(())
}

/// 只用于注入保存失败、记录调用次序；正常持久化由上面的真实数据库测试验证。
#[derive(Default)]
struct RecordingRepository {
    attempted: Mutex<Vec<File>>,
    fail_save: bool,
}

#[async_trait]
impl FileRepository for RecordingRepository {
    async fn save(&self, file: File) -> Result<()> {
        self.attempted.lock().await.push(file);
        if self.fail_save {
            bail!("injected database save failure");
        }
        Ok(())
    }

    async fn get_by_id(&self, file_id: &str) -> Result<Option<File>> {
        Ok(self
            .attempted
            .lock()
            .await
            .iter()
            .find(|file| file.id == file_id)
            .cloned())
    }
}

#[tokio::test]
async fn removes_uploaded_object_when_saving_metadata_fails() -> Result<()> {
    let storage = Arc::new(Storage::single(drivers::mem::new()));
    let repository = Arc::new(RecordingRepository {
        fail_save: true,
        ..Default::default()
    });
    let service = LocoFileStorage::new(storage.clone(), repository.clone());
    let error = service
        .upload_file(upload("report.txt", b"content"))
        .await
        .unwrap_err();
    assert!(format!("{error:#}").contains("injected database save failure"));
    let attempted = repository.attempted.lock().await;
    assert_eq!(attempted.len(), 1);
    assert!(storage
        .download::<Vec<u8>>(Path::new(&attempted[0].key))
        .await
        .is_err());
    Ok(())
}

#[tokio::test]
async fn does_not_save_metadata_when_storage_upload_fails() {
    let repository = Arc::new(RecordingRepository::default());
    let service = LocoFileStorage::new(
        Arc::new(Storage::single(drivers::null::new())),
        repository.clone(),
    );
    assert!(service
        .upload_file(upload("report.txt", b"content"))
        .await
        .is_err());
    assert!(repository.attempted.lock().await.is_empty());
}

#[tokio::test]
async fn validates_names_and_preserves_empty_and_extensionless_files() -> Result<()> {
    let storage = Arc::new(Storage::single(drivers::mem::new()));
    let repository = Arc::new(RecordingRepository::default());
    let service = LocoFileStorage::new(storage, repository.clone());
    for filename in ["", " ", ".", "..", "../", "bad\nname"] {
        let error = service
            .upload_file(upload(filename, b"data"))
            .await
            .unwrap_err();
        assert!(error.is::<InvalidFileUpload>());
    }
    assert!(service
        .upload_file(upload(&"a".repeat(256), b"data"))
        .await
        .is_err());
    let mut bad_mime = upload("report", b"data");
    bad_mime.mime_type = Some("text/plain\r\ninvalid".to_string());
    assert!(service.upload_file(bad_mime).await.is_err());
    assert!(repository.attempted.lock().await.is_empty());

    let file = service.upload_file(upload("../../README", b"")).await?;
    assert_eq!(file.filename, "README");
    assert_eq!(file.extension, "");
    assert_eq!(file.mime_type, "");
    assert_eq!(file.size, 0);
    assert!(!file.key.contains(".."));
    let (stream, _) = service.download_file(&file.id).await?;
    assert!(read_all(stream).await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn preserves_dotted_extensions_and_checks_the_complete_key_length() -> Result<()> {
    let storage = Arc::new(Storage::single(drivers::mem::new()));
    let repository = Arc::new(RecordingRepository::default());
    let service = LocoFileStorage::new(storage.clone(), repository.clone());

    for (filename, extension) in [
        (".env", ""),
        ("..hidden", ""),
        ("archive.tar.gz", ".gz"),
        ("trailing.", "."),
        ("报告.文档", ".文档"),
    ] {
        let file = service.upload_file(upload(filename, b"content")).await?;
        assert_eq!(file.extension, extension);
        assert!(file.key.ends_with(&format!("{}{extension}", file.id)));
        assert_eq!(
            storage.download::<Vec<u8>>(Path::new(&file.key)).await?,
            b"content"
        );
    }

    // 文件名未超长，但日期 + UUID + 扩展名已超过 key 列的上限；上传前就应拒绝。
    let attempted_before = repository.attempted.lock().await.len();
    let error = service
        .upload_file(upload(&format!("a.{}", "x".repeat(208)), b"data"))
        .await
        .unwrap_err();
    assert!(error.is::<InvalidFileUpload>());
    assert_eq!(repository.attempted.lock().await.len(), attempted_before);

    let file = service
        .upload_file(upload(&format!("a.{}", "x".repeat(207)), b"data"))
        .await?;
    assert_eq!(file.key.chars().count(), 255);
    Ok(())
}

#[tokio::test]
async fn downloads_existing_files_with_legacy_keys() -> Result<()> {
    let storage = Arc::new(Storage::single(drivers::mem::new()));
    let repository = Arc::new(RecordingRepository::default());
    let service = LocoFileStorage::new(storage.clone(), repository.clone());
    let mut legacy = File {
        filename: "legacy.txt".to_string(),
        extension: "txt".to_string(),
        mime_type: "text/plain".to_string(),
        size: 6,
        ..Default::default()
    };
    legacy.key = format!("uploads/{}", legacy.id);
    storage
        .upload(Path::new(&legacy.key), &Bytes::from_static(b"legacy"))
        .await?;
    repository.save(legacy.clone()).await?;

    let (stream, file) = service.download_file(&legacy.id).await?;
    assert_eq!(file, legacy);
    assert_eq!(read_all(stream).await?, b"legacy");
    Ok(())
}
