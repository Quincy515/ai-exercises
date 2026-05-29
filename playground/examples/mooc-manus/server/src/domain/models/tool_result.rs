use serde::{Deserialize, Serialize};

/// 工具结果 Domain 模型。
/// Domain model for tool execution results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ToolResult<T> {
    /// 是否成功调用。
    /// Whether the tool call succeeded.
    pub success: bool,
    /// 额外的信息提示。
    /// Extra message for the caller.
    pub message: Option<String>,
    /// 工具的执行结果 / 数据。
    /// Tool execution result data.
    pub data: Option<T>,
}

impl<T> Default for ToolResult<T> {
    fn default() -> Self {
        Self {
            success: true,
            message: Some(String::new()),
            data: None,
        }
    }
}
