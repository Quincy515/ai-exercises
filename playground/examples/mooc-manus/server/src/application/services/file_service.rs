use std::sync::Arc;

use anyhow::{Context, Result};

use crate::domain::{
    external::{FileNotFound, FileStorage, FileStream, UploadFile},
    models::File,
    repositories::FileRepository,
};

/// MoocManus 文件系统服务。
pub struct FileService {
    file_storage: Arc<dyn FileStorage>,
    file_repository: Arc<dyn FileRepository>,
}

impl FileService {
    /// 构造函数，完成文件服务的初始化。
    pub fn new(
        file_storage: Arc<dyn FileStorage>,
        file_repository: Arc<dyn FileRepository>,
    ) -> Self {
        Self {
            file_storage,
            file_repository,
        }
    }

    /// 将传递的文件上传到对象存储并记录上传数据。
    pub async fn upload_file(&self, upload_file: UploadFile) -> Result<File> {
        self.file_storage.upload_file(upload_file).await
    }

    /// 根据传递的文件 id 获取文件信息。
    pub async fn get_file_info(&self, file_id: &str) -> Result<File> {
        self.file_repository
            .get_by_id(file_id)
            .await
            .context("获取文件信息失败")?
            .ok_or_else(|| FileNotFound::new(file_id).into())
    }

    /// 根据传递的文件 id 下载文件。
    pub async fn download_file(&self, file_id: &str) -> Result<(FileStream, File)> {
        self.file_storage.download_file(file_id).await
    }
}
