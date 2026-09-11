use serde::Serialize;
use utoipa::ToSchema;

use crate::domain::models::File;

/// 上传接口的 multipart 文档模型，实际请求由 Axum Multipart 提取器读取。
#[derive(ToSchema)]
pub struct FileUploadRequest {
    /// 要上传的单个文件。
    #[schema(value_type = String, format = Binary)]
    pub file: Vec<u8>,
}

/// 文件基础信息；HTTP 表达与数据库字段分开维护。
#[derive(Debug, Serialize, ToSchema)]
pub struct FileInfoResponse {
    /// 文件业务 UUID。
    pub id: String,
    pub filename: String,
    /// 沙箱路径，尚未同步到沙箱时为空。
    pub filepath: String,
    /// 对象存储 key。
    pub key: String,
    /// 包含前导点的扩展名，如 .pdf；没有扩展名时为空。
    pub extension: String,
    pub mime_type: String,
    /// 文件大小，单位为字节。
    pub size: usize,
}

impl From<File> for FileInfoResponse {
    fn from(file: File) -> Self {
        Self {
            id: file.id,
            filename: file.filename,
            filepath: file.filepath,
            key: file.key,
            extension: file.extension,
            mime_type: file.mime_type,
            size: file.size,
        }
    }
}

/// 文件上传、获取信息接口的成功响应。
#[derive(Debug, Serialize, ToSchema)]
pub struct FileResponse {
    pub code: u16,
    pub msg: String,
    pub data: FileInfoResponse,
}

impl FileResponse {
    pub fn success(msg: impl Into<String>, file: File) -> Self {
        Self {
            code: 200,
            msg: msg.into(),
            data: file.into(),
        }
    }
}
