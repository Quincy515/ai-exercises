use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::domain::models::{Plan, Step};

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
    pub attachments: Vec<Value>,
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

/// 工具事件
/// Tool event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolEvent {
    #[serde(flatten, default)]
    pub base: BaseEvent,
}

impl Default for ToolEvent {
    fn default() -> Self {
        Self {
            base: BaseEvent {
                event_type: EventType::Tool,
                ..BaseEvent::default()
            },
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
    use super::{BaseEvent, Event, EventType, MessageEvent, MessageRole, PlanEvent, TitleEvent};
    use serde_json::Value;
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
    fn event_type_uses_python_literal_values() {
        assert_eq!(serde_json::to_string(&EventType::Plan).unwrap(), "\"plan\"");
        assert_eq!(serde_json::to_string(&EventType::Empty).unwrap(), "\"\"");
    }
}
