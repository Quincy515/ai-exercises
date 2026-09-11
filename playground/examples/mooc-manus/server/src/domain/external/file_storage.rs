use std::{io, pin::Pin};

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;

use crate::domain::models::File;

/// 上传文件源；由接口层读取 multipart 后构建，领域层不依赖 HTTP 提取器。
pub struct UploadFile {
    /// 原始文件名。
    pub filename: String,
    /// 文件 MIME 类型；未提供时元数据记录为空，由 HTTP 层选择下载响应的默认类型。
    pub mime_type: Option<String>,
    /// 文件二进制内容，大小由实际字节数计算。
    pub content: Bytes,
}

/// 异步文件源；逐块读取，读取过程中的错误也通过流返回。
pub type FileStream = Pin<Box<dyn Stream<Item = io::Result<Bytes>> + Send>>;

/// 文件记录不存在；接口层可识别该类型并映射为 404，不混淆数据库读取失败。
#[derive(Debug, thiserror::Error)]
#[error("该文件[{file_id}]不存在")]
pub struct FileNotFound {
    pub file_id: String,
}

impl FileNotFound {
    pub fn new(file_id: impl Into<String>) -> Self {
        Self {
            file_id: file_id.into(),
        }
    }
}

/// 上传元数据不合法；接口层可识别该类型并映射为 400。
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct InvalidFileUpload(pub &'static str);

/// 文件存储桶协议。
#[async_trait]
pub trait FileStorage: Send + Sync {
    /// 根据传递的文件源上传文件后返回文件信息。
    async fn upload_file(&self, upload_file: UploadFile) -> Result<File>;

    /// 根据传递的文件 id 下载文件，并返回文件源 + 文件信息。
    async fn download_file(&self, file_id: &str) -> Result<(FileStream, File)>;
}
