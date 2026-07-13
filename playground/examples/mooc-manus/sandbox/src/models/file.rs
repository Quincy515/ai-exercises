use serde::Serialize;
use utoipa::ToSchema;

/// 文件读取结果
#[derive(Debug, Clone, Default, Serialize, ToSchema, PartialEq, Eq)]
pub struct FileReadResult {
    /// 要读取的文件路径
    pub filepath: String,
    /// 读取的文件内容
    pub content: String,
}
