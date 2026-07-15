use serde::Deserialize;
use utoipa::{IntoParams, ToSchema};

/// 读取文件请求结构体
#[derive(Debug, Deserialize, ToSchema, PartialEq, Eq)]
pub struct FileReadRequest {
    /// 要读取文件的绝对路径
    pub file_path: String,
    /// (可选)读取的起始行, 索引从0开始
    pub start_line: Option<usize>,
    /// (可选)结束行号, 不包含该行
    pub end_line: Option<usize>,
    /// (可选)是否使用sudo权限
    #[serde(default = "default_false")]
    pub sudo: Option<bool>,
    /// (可选)要返回的内容的最大长度
    #[serde(default = "default_max_length")]
    pub max_length: Option<usize>,
}

const fn default_false() -> Option<bool> {
    Some(false)
}

const fn default_max_length() -> Option<usize> {
    Some(10_000)
}

/// 写入文件请求结构体
#[derive(Debug, Deserialize, ToSchema, PartialEq, Eq)]
pub struct FileWriteRequest {
    /// 要写入文件的绝对路径
    pub file_path: String,
    /// 要写入的文本内容
    pub content: String,
    /// (可选)是否使用追加模式
    #[serde(default = "default_false")]
    pub append: Option<bool>,
    /// (可选)是否在内容开头添加前置空行
    #[serde(default = "default_false")]
    pub leading_newline: Option<bool>,
    /// (可选)是否在内容结尾添加后置空行
    #[serde(default = "default_false")]
    pub trailing_newline: Option<bool>,
    /// (可选)是否使用sudo权限
    #[serde(default = "default_false")]
    pub sudo: Option<bool>,
}

/// 查找替换文件内容请求结构体
#[derive(Debug, Deserialize, ToSchema, PartialEq, Eq)]
pub struct FileReplaceRequest {
    /// 要替换内容的文件绝对路径
    pub file_path: String,
    /// 要替换的原始字符串
    pub old_str: String,
    /// 要替换的新字符串
    pub new_str: String,
    /// (可选)是否使用sudo权限
    #[serde(default = "default_false")]
    pub sudo: Option<bool>,
}

/// 文件内容查找请求结构体
#[derive(Debug, Deserialize, ToSchema, PartialEq, Eq)]
pub struct FileSearchRequest {
    /// 要查找内容的文件绝对路径
    pub file_path: String,
    /// 搜索正则表达式
    pub regex: String,
    /// (可选)是否使用sudo权限
    #[serde(default = "default_false")]
    pub sudo: Option<bool>,
}

/// 文件查找请求结构体
#[derive(Debug, Deserialize, ToSchema, PartialEq, Eq)]
pub struct FileFindRequest {
    /// 搜索的目录绝对路径
    pub dir_path: String,
    /// 文件名模式，使用glob语法
    pub glob_pattern: String,
}

/// 文件上传表单结构体，仅用于生成OpenAPI文档
#[derive(Debug, ToSchema)]
pub struct FileUploadForm {
    /// 上传的文件源
    #[schema(
        value_type = String,
        format = Binary,
        content_media_type = "application/octet-stream"
    )]
    pub file: Vec<u8>,
    /// 上传的文件绝对路径，默认使用/tmp目录和原文件名
    pub file_path: Option<String>,
}

/// 文件下载查询参数
#[derive(Debug, Deserialize, IntoParams, PartialEq, Eq)]
#[into_params(parameter_in = Query)]
pub struct FileDownloadRequest {
    /// 要下载的文件绝对路径
    pub file_path: String,
}

/// 检查文件是否存在请求结构体
#[derive(Debug, Deserialize, ToSchema, PartialEq, Eq)]
pub struct FileCheckRequest {
    /// 要检查是否存在的文件绝对路径
    pub file_path: String,
}

/// 删除文件请求结构体
#[derive(Debug, Deserialize, ToSchema, PartialEq, Eq)]
pub struct FileDeleteRequest {
    /// 要删除的文件绝对路径
    pub file_path: String,
}

#[cfg(test)]
mod tests {
    use serde::de::value::{Error, MapDeserializer};

    use super::*;

    #[test]
    fn request_uses_rust_field_name_and_defaults() {
        let request = FileReadRequest::deserialize(MapDeserializer::<_, Error>::new(
            [("file_path", "/tmp/agent.txt")].into_iter(),
        ))
        .expect("minimal request should be deserializable");

        assert_eq!(request.file_path, "/tmp/agent.txt");
        assert_eq!(request.start_line, None);
        assert_eq!(request.end_line, None);
        assert_eq!(request.sudo, Some(false));
        assert_eq!(request.max_length, Some(10_000));
    }

    #[test]
    fn write_request_defaults_to_overwrite_without_extra_newlines() {
        let request = FileWriteRequest::deserialize(MapDeserializer::<_, Error>::new(
            [
                ("file_path", "/tmp/agent.txt"),
                ("content", "Agent写入文件"),
            ]
            .into_iter(),
        ))
        .expect("minimal write request should be deserializable");

        assert_eq!(request.file_path, "/tmp/agent.txt");
        assert_eq!(request.content, "Agent写入文件");
        assert_eq!(request.append, Some(false));
        assert_eq!(request.leading_newline, Some(false));
        assert_eq!(request.trailing_newline, Some(false));
        assert_eq!(request.sudo, Some(false));
    }

    #[test]
    fn replace_and_search_requests_default_to_regular_permissions() {
        let replace = FileReplaceRequest::deserialize(MapDeserializer::<_, Error>::new(
            [
                ("file_path", "/tmp/agent.txt"),
                ("old_str", "旧内容"),
                ("new_str", "新内容"),
            ]
            .into_iter(),
        ))
        .expect("minimal replace request should be deserializable");
        let search = FileSearchRequest::deserialize(MapDeserializer::<_, Error>::new(
            [("file_path", "/tmp/agent.txt"), ("regex", "^Agent")].into_iter(),
        ))
        .expect("minimal search request should be deserializable");

        assert_eq!(replace.sudo, Some(false));
        assert_eq!(search.sudo, Some(false));
    }
}
