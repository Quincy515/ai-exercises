use serde::Deserialize;
use utoipa::ToSchema;

/// 读取文件请求结构体
#[derive(Debug, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ReadFileRequest {
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
pub struct WriteFileRequest {
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

#[cfg(test)]
mod tests {
    use serde::de::value::{Error, MapDeserializer};

    use super::*;

    #[test]
    fn request_uses_rust_field_name_and_defaults() {
        let request = ReadFileRequest::deserialize(MapDeserializer::<_, Error>::new(
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
        let request = WriteFileRequest::deserialize(MapDeserializer::<_, Error>::new(
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
}
