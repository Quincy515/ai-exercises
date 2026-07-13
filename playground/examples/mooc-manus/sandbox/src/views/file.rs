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
    #[serde(default = "default_sudo")]
    pub sudo: Option<bool>,
    /// (可选)要返回的内容的最大长度
    #[serde(default = "default_max_length")]
    pub max_length: Option<usize>,
}

const fn default_sudo() -> Option<bool> {
    Some(false)
}

const fn default_max_length() -> Option<usize> {
    Some(10_000)
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
}
