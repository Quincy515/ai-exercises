use serde::{Deserialize, Serialize};

///文件信息 Domian 模型，用于记录 Manus/Human 上传 or 生成的文件
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct File {
    /// 文件 ID
    pub id: String,
    /// 文件名
    pub filename: String,
    /// 文件路径
    pub filepath: String,
    /// 文件云存储中的路径
    pub key: String,
    /// 文件扩展名
    pub extension: String,
    /// 文件 mime-type 类型
    pub mime_type: String,
    /// 文件大小，单位为字节
    pub size: usize,
}

impl Default for File {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            filename: String::new(),
            filepath: String::new(),
            key: String::new(),
            extension: String::new(),
            mime_type: String::new(),
            size: 0,
        }
    }
}
