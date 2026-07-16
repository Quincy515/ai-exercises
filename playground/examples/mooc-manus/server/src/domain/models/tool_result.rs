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

impl<T> ToolResult<T> {
    /// 将从沙箱中返回的 API 数据转换成工具结果。
    pub fn from_sandbox(code: i32, msg: impl Into<String>, data: Option<T>) -> Self {
        Self {
            success: code < 300,
            message: Some(msg.into()),
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ToolResult;

    #[test]
    fn converts_successful_sandbox_response() {
        let result = ToolResult::from_sandbox(200, "执行成功", Some("output"));

        assert!(result.success);
        assert_eq!(result.message.as_deref(), Some("执行成功"));
        assert_eq!(result.data, Some("output"));
    }

    #[test]
    fn converts_failed_sandbox_response() {
        let result = ToolResult::<String>::from_sandbox(300, "执行失败", None);

        assert!(!result.success);
        assert_eq!(result.message.as_deref(), Some("执行失败"));
        assert_eq!(result.data, None);
    }
}
