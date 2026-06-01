use serde::{Deserialize, Serialize};

/// 用户传递的消息
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Message {
    /// 用户发送的消息
    pub message: String,
    /// 用户发送的附件
    pub attachments: Vec<String>,
}
