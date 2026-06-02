//! 定义规划 Agent 使用的系统提示词和任务模板。

/// 规划 Agent 系统预设 prompt。
pub const PLANNER_SYSTEM_PROMPT: &str = r#""#;

/// 创建 Plan 规划提示词模板，内部有 message 和 attachments 占位符。
pub const CREATE_PLAN_PROMPT: &str = "{message}\n{attachments}";

/// 更新 Plan 规划提示词模板，内部有 plan 和 step 占位符。
pub const UPDATE_PLAN_PROMPT: &str = "{plan}\n{step}";
