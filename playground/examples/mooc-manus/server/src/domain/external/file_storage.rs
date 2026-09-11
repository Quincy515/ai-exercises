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
    /// 文件 MIME 类型；未提供时使用 application/octet-stream。
    pub mime_type: Option<String>,
    /// 文件二进制内容，大小由实际字节数计算。
    pub content: Bytes,
}

/// 异步文件源；逐块读取，读取过程中的错误也通过流返回。
pub type FileStream = Pin<Box<dyn Stream<Item = io::Result<Bytes>> + Send>>;

/// 文件存储桶协议。
#[async_trait]
pub trait FileStorage: Send + Sync {
    /// 根据传递的文件源上传文件后返回文件信息。
    async fn upload_file(&self, upload_file: UploadFile) -> Result<File>;

    /// 根据传递的文件 id 下载文件，并返回文件源 + 文件信息。
    async fn download_file(&self, file_id: &str) -> Result<(FileStream, File)>;
}
