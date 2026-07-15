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

/// 文件内容替换结果
#[derive(Debug, Clone, Default, Serialize, ToSchema, PartialEq, Eq)]
pub struct FileReplaceResult {
    /// 要替换内容的文件绝对路径
    pub file_path: String,
    /// 替换内容的次数
    pub replaced_count: usize,
}

/// 文件内容搜索结果
#[derive(Debug, Clone, Default, Serialize, ToSchema, PartialEq, Eq)]
pub struct FileSearchResult {
    /// 要搜索内容的文件绝对路径
    pub file_path: String,
    /// 匹配内容列表
    pub matches: Vec<String>,
    /// 匹配内容的零基行号列表
    pub line_numbers: Vec<usize>,
}

/// 文件查找结果
#[derive(Debug, Clone, Default, Serialize, ToSchema, PartialEq, Eq)]
pub struct FileFindResult {
    /// 搜索的目录绝对路径
    pub dir_path: String,
    /// 检索到的文件列表
    pub files: Vec<String>,
}

/// 文件上传结果
#[derive(Debug, Clone, Default, Serialize, ToSchema, PartialEq, Eq)]
pub struct FileUploadResult {
    /// 上传文件的绝对路径
    pub file_path: String,
    /// 上传文件的大小，单位为字节
    pub file_size: u64,
    /// 文件是否上传成功
    pub success: bool,
}

/// 文件存在检查结果
#[derive(Debug, Clone, Default, Serialize, ToSchema, PartialEq, Eq)]
pub struct FileCheckResult {
    /// 需要检查的文件绝对路径
    pub file_path: String,
    /// 文件是否存在
    pub exists: bool,
}

/// 文件删除结果
#[derive(Debug, Clone, Default, Serialize, ToSchema, PartialEq, Eq)]
pub struct FileDeleteResult {
    /// 需要删除的文件绝对路径
    pub file_path: String,
    /// 文件是否删除成功
    pub deleted: bool,
}
