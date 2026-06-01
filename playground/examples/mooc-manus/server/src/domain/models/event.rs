use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::domain::models::{File, Plan, Step, ToolResult};

/// 事件类型
/// Event type.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EventType {
    /// 基础事件的空类型
    /// Empty type for base events.
    #[default]
    #[serde(rename = "")]
    Empty,
    /// 规划事件
    /// Plan event.
    Plan,
    /// 标题事件
    /// Title event.
    Title,
    /// 步骤事件
    /// Step event.
    Step,
    /// 消息事件
    /// Message event.
    Message,
    /// 工具事件
    /// Tool event.
    Tool,
    /// 等待事件
    /// Wait event.
    Wait,
    /// 错误事件
    /// Error event.
    Error,
    /// 结束事件
    /// Done event.
    Done,
}

/// 规划事件状态
/// Plan event status.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PlanEventStatus {
    /// 已创建
    /// Created.
    #[default]
    Created,
    /// 已更新
    /// Updated.
    Updated,
    /// 已完成
    /// Completed.
    Completed,
}

/// 步骤事件状态
/// Step event status.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum StepEventStatus {
    /// 已开始
    /// Started.
    #[default]
    Started,
    /// 已完成
    /// Completed.
    Completed,
    /// 失败
    /// Failed.
    Failed,
}

/// 工具事件状态
/// Tool event status.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolEventStatus {
    /// 调用中
    /// Calling.
    #[default]
    Calling,
    /// 调用完毕
    /// Called.
    Called,
}

/// 消息角色
/// Message role.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// 用户消息
    /// User message.
    User,
    /// AI 消息
    /// Assistant message.
    #[default]
    Assistant,
}

/// 基础事件字段
/// Common fields shared by all events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BaseEvent {
    /// 事件 id
    /// Event id.
    #[serde(default = "new_event_id")]
    pub id: String,
    /// 事件的类型
    /// Event type.
    #[serde(rename = "type", default)]
    pub event_type: EventType,
    /// 事件创建时间
    /// Event creation time.
    #[serde(default = "event_now")]
    pub created_at: DateTime<Utc>,
}

impl Default for BaseEvent {
    fn default() -> Self {
        Self {
            id: new_event_id(),
            event_type: EventType::Empty,
            created_at: event_now(),
        }
    }
}

/// 规划事件类型
/// Plan event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanEvent {
    #[serde(flatten, default)]
    pub base: BaseEvent,
    /// 规划
    /// Plan.
    pub plan: Plan,
    /// 规划事件状态
    /// Plan event status.
    #[serde(default)]
    pub status: PlanEventStatus,
}

impl Default for PlanEvent {
    fn default() -> Self {
        Self {
            base: BaseEvent {
                event_type: EventType::Plan,
                ..BaseEvent::default()
            },
            plan: Plan::default(),
            status: PlanEventStatus::Created,
        }
    }
}

/// 标题事件类型
/// Title event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TitleEvent {
    #[serde(flatten, default)]
    pub base: BaseEvent,
    /// 标题
    /// Title.
    #[serde(default)]
    pub title: String,
}

impl Default for TitleEvent {
    fn default() -> Self {
        Self {
            base: BaseEvent {
                event_type: EventType::Title,
                ..BaseEvent::default()
            },
            title: String::new(),
        }
    }
}

/// 子任务 / 步骤事件
/// Step event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StepEvent {
    #[serde(flatten, default)]
    pub base: BaseEvent,
    /// 步骤信息
    /// Step information.
    pub step: Step,
    /// 步骤状态
    /// Step status.
    #[serde(default)]
    pub status: StepEventStatus,
}

impl Default for StepEvent {
    fn default() -> Self {
        Self {
            base: BaseEvent {
                event_type: EventType::Step,
                ..BaseEvent::default()
            },
            step: Step::default(),
            status: StepEventStatus::Started,
        }
    }
}

/// 消息事件，包含人类消息和 AI 消息
/// Message event for user and assistant messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessageEvent {
    #[serde(flatten, default)]
    pub base: BaseEvent,
    /// 消息角色
    /// Message role.
    #[serde(default)]
    pub role: MessageRole,
    /// 消息本身
    /// Message text.
    #[serde(default)]
    pub message: String,
    /// 附件列表信息
    /// Attachment list.
    #[serde(default)]
    pub attachments: Vec<File>,
}

impl Default for MessageEvent {
    fn default() -> Self {
        Self {
            base: BaseEvent {
                event_type: EventType::Message,
                ..BaseEvent::default()
            },
            role: MessageRole::Assistant,
            message: String::new(),
            attachments: Vec::new(),
        }
    }
}

/// 工具内容
/// Tool content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BrowserToolContent {
    /// 浏览器快照截图
    /// Browser screenshot.
    pub screenshot: String,
}

/// MCP 工具内容
/// MCP tool content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpToolContent {
    /// MCP 工具返回结果
    /// MCP tool result.
    pub result: Value,
}

/// 工具扩展内容
/// Extended tool content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ToolContent {
    Browser(BrowserToolContent),
    Mcp(McpToolContent),
}

/// 工具事件
/// Tool event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolEvent {
    #[serde(flatten, default)]
    pub base: BaseEvent,
    /// 工具调用 id
    /// Tool call id.
    pub tool_call_id: String,
    /// 工具集的名字
    /// Tool collection name.
    pub tool_name: String,
    /// 工具扩展内容
    /// Extended tool content.
    pub tool_content: Option<ToolContent>,
    /// LLM 调用的函数 / 工具名字
    /// LLM-called function or tool name.
    pub function_name: String,
    /// LLM 生成的工具调用参数
    /// LLM-generated tool arguments.
    pub function_args: Map<String, Value>,
    /// 工具调用结果
    /// Tool call result.
    pub function_result: Option<ToolResult<Value>>,
    /// 工具事件状态
    /// Tool event status.
    #[serde(default)]
    pub status: ToolEventStatus,
}

impl Default for ToolEvent {
    fn default() -> Self {
        Self {
            base: BaseEvent {
                event_type: EventType::Tool,
                ..BaseEvent::default()
            },
            tool_call_id: String::new(),
            tool_name: String::new(),
            tool_content: None,
            function_name: String::new(),
            function_args: Map::new(),
            function_result: None,
            status: ToolEventStatus::Calling,
        }
    }
}

/// 等待事件，等待用户输入确认
/// Wait event for user confirmation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WaitEvent {
    #[serde(flatten, default)]
    pub base: BaseEvent,
}

impl Default for WaitEvent {
    fn default() -> Self {
        Self {
            base: BaseEvent {
                event_type: EventType::Wait,
                ..BaseEvent::default()
            },
        }
    }
}

/// 错误事件
/// Error event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorEvent {
    #[serde(flatten, default)]
    pub base: BaseEvent,
    /// 错误信息
    /// Error message.
    #[serde(default)]
    pub error: String,
}

impl Default for ErrorEvent {
    fn default() -> Self {
        Self {
            base: BaseEvent {
                event_type: EventType::Error,
                ..BaseEvent::default()
            },
            error: String::new(),
        }
    }
}

/// 结束事件类型
/// Done event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DoneEvent {
    #[serde(flatten, default)]
    pub base: BaseEvent,
}

impl Default for DoneEvent {
    fn default() -> Self {
        Self {
            base: BaseEvent {
                event_type: EventType::Done,
                ..BaseEvent::default()
            },
        }
    }
}

/// 应用事件类型声明
/// Application event union.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum Event {
    Plan(PlanEvent),
    Title(TitleEvent),
    Step(StepEvent),
    Message(MessageEvent),
    Tool(ToolEvent),
    Wait(WaitEvent),
    Error(ErrorEvent),
    Done(DoneEvent),
}

fn new_event_id() -> String {
    Uuid::new_v4().to_string()
}

fn event_now() -> DateTime<Utc> {
    Utc::now()
}

#[cfg(test)]
mod tests {
    use super::{
        BaseEvent, BrowserToolContent, Event, EventType, McpToolContent, MessageEvent, MessageRole,
        PlanEvent, TitleEvent, ToolContent, ToolEvent, ToolEventStatus,
    };
    use serde_json::{json, Value};
    use uuid::Uuid;

    #[test]
    fn base_event_matches_python_default_type() {
        let value = serde_json::to_value(BaseEvent::default()).unwrap();

        assert_eq!(value.get("type"), Some(&Value::String(String::new())));
        assert!(value.get("created_at").is_some());

        let id = value.get("id").and_then(Value::as_str).unwrap();
        assert!(Uuid::parse_str(id).is_ok());
    }

    #[test]
    fn plan_event_matches_python_json_shape() {
        let event = PlanEvent::default();
        let value = serde_json::to_value(&event).unwrap();

        assert_eq!(value.get("type"), Some(&Value::String("plan".to_string())));
        assert_eq!(
            value.get("status"),
            Some(&Value::String("created".to_string()))
        );
        assert!(value.get("plan").is_some());
        assert!(value.get("created_at").is_some());
        assert!(value.get("base").is_none());

        let id = value.get("id").and_then(Value::as_str).unwrap();
        assert!(Uuid::parse_str(id).is_ok());
    }

    #[test]
    fn message_event_defaults_match_python() {
        let value = serde_json::to_value(MessageEvent::default()).unwrap();

        assert_eq!(
            value.get("type"),
            Some(&Value::String("message".to_string()))
        );
        assert_eq!(
            value.get("role"),
            Some(&Value::String("assistant".to_string()))
        );
        assert_eq!(value.get("message"), Some(&Value::String(String::new())));
        assert_eq!(value.get("attachments"), Some(&Value::Array(Vec::new())));
    }

    #[test]
    fn event_union_uses_literal_type_to_select_variant() {
        let event = Event::Title(TitleEvent::default());
        let value = serde_json::to_value(event).unwrap();

        assert_eq!(value.get("type"), Some(&Value::String("title".to_string())));
    }

    #[test]
    fn message_role_uses_python_literal_values() {
        assert_eq!(
            serde_json::to_string(&MessageRole::User).unwrap(),
            "\"user\""
        );
        assert_eq!(
            serde_json::to_string(&MessageRole::Assistant).unwrap(),
            "\"assistant\""
        );
    }

    #[test]
    fn tool_event_defaults_match_python() {
        let value = serde_json::to_value(ToolEvent::default()).unwrap();

        assert_eq!(value.get("type"), Some(&Value::String("tool".to_string())));
        assert_eq!(
            value.get("tool_call_id"),
            Some(&Value::String(String::new()))
        );
        assert_eq!(value.get("tool_name"), Some(&Value::String(String::new())));
        assert_eq!(value.get("tool_content"), Some(&Value::Null));
        assert_eq!(
            value.get("function_name"),
            Some(&Value::String(String::new()))
        );
        assert_eq!(value.get("function_args"), Some(&json!({})));
        assert_eq!(value.get("function_result"), Some(&Value::Null));
        assert_eq!(
            value.get("status"),
            Some(&Value::String("calling".to_string()))
        );
    }

    #[test]
    fn tool_content_matches_python_union_shape() {
        let browser = ToolContent::Browser(BrowserToolContent {
            screenshot: "snapshot.png".to_string(),
        });
        assert_eq!(
            serde_json::to_value(browser).unwrap(),
            json!({ "screenshot": "snapshot.png" })
        );

        let mcp = ToolContent::Mcp(McpToolContent {
            result: json!({ "answer": 42 }),
        });
        assert_eq!(
            serde_json::to_value(mcp).unwrap(),
            json!({ "result": { "answer": 42 } })
        );
    }

    #[test]
    fn tool_event_status_uses_python_enum_values() {
        assert_eq!(
            serde_json::to_string(&ToolEventStatus::Calling).unwrap(),
            "\"calling\""
        );
        assert_eq!(
            serde_json::to_string(&ToolEventStatus::Called).unwrap(),
            "\"called\""
        );
    }

    #[test]
    fn event_type_uses_python_literal_values() {
        assert_eq!(serde_json::to_string(&EventType::Plan).unwrap(), "\"plan\"");
        assert_eq!(serde_json::to_string(&EventType::Empty).unwrap(), "\"\"");
    }
}
