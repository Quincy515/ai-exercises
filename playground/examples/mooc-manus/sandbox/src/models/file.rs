use serde::Serialize;
use utoipa::ToSchema;

/// 文件读取结果
#[derive(Debug, Clone, Default, Serialize, ToSchema, PartialEq, Eq)]
pub struct FileReadResult {
    /// 要读取的文件绝对路径
    pub file_path: String,
    /// 读取的文件内容
    pub content: String,
}

/// 文件写入结果
#[derive(Debug, Clone, Default, Serialize, ToSchema, PartialEq, Eq)]
pub struct FileWriteResult {
    /// 要写入的文件绝对路径
    pub file_path: String,
    /// 写入文件内容的字节数
    pub bytes_written: Option<usize>,
}
