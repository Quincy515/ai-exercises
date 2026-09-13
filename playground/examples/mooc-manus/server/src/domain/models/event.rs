use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::domain::models::{File, Plan, SearchResultItem, Step, ToolResult};

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

/// 搜索工具内容
/// Search tool content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchToolContent {
    /// 搜索结果列表
    /// Search result list.
    pub results: Vec<SearchResultItem>,
}

/// Shell 工具内容
/// Shell tool content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShellToolContent {
    /// 控制台内容
    /// Console content.
    pub console: Value,
}

/// 文件工具内容
/// File tool content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileToolContent {
    /// 文件内容
    /// File content.
    pub content: String,
}

/// MCP 工具内容
/// MCP tool content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpToolContent {
    /// MCP 工具返回结果
    /// MCP tool result.
    pub result: Value,
}

/// A2A 智能体工具内容
/// A2A agent tool content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct A2aToolContent {
    /// A2A 智能体调用结果
    /// A2A agent invocation result.
    pub a2a_result: Value,
}

/// 工具扩展内容
/// Extended tool content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ToolContent {
    Browser(BrowserToolContent),
    Search(SearchToolContent),
    Shell(ShellToolContent),
    File(FileToolContent),
    Mcp(McpToolContent),
    A2a(A2aToolContent),
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
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl Event {
    /// 使用消息队列返回的 id 更新事件，便于关联队列消息和会话历史。
    pub fn set_id(&mut self, id: impl Into<String>) {
        let base = match self {
            Self::Plan(event) => &mut event.base,
            Self::Title(event) => &mut event.base,
            Self::Step(event) => &mut event.base,
            Self::Message(event) => &mut event.base,
            Self::Tool(event) => &mut event.base,
            Self::Wait(event) => &mut event.base,
            Self::Error(event) => &mut event.base,
            Self::Done(event) => &mut event.base,
        };
        base.id = id.into();
    }
}

impl Serialize for Event {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // 保持原有扁平 JSON，同时防止公共 base 字段与 Rust 枚举变体不一致。
        // 否则保存的是一种事件，重新读取时却会变成另一种事件。
        match self {
            Self::Plan(event) => {
                serialize_typed_event(event, event.base.event_type, EventType::Plan, serializer)
            }
            Self::Title(event) => {
                serialize_typed_event(event, event.base.event_type, EventType::Title, serializer)
            }
            Self::Step(event) => {
                serialize_typed_event(event, event.base.event_type, EventType::Step, serializer)
            }
            Self::Message(event) => {
                serialize_typed_event(event, event.base.event_type, EventType::Message, serializer)
            }
            Self::Tool(event) => {
                serialize_typed_event(event, event.base.event_type, EventType::Tool, serializer)
            }
            Self::Wait(event) => {
                serialize_typed_event(event, event.base.event_type, EventType::Wait, serializer)
            }
            Self::Error(event) => {
                serialize_typed_event(event, event.base.event_type, EventType::Error, serializer)
            }
            Self::Done(event) => {
                serialize_typed_event(event, event.base.event_type, EventType::Done, serializer)
            }
        }
    }
}

/// 校验具体事件与类型标签，再复用该事件原有的序列化实现。
fn serialize_typed_event<T, S>(
    event: &T,
    actual_type: EventType,
    expected_type: EventType,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    T: Serialize,
    S: Serializer,
{
    if actual_type != expected_type {
        return Err(serde::ser::Error::custom(format!(
            "event type mismatch: expected {expected_type:?}, found {actual_type:?}"
        )));
    }
    event.serialize(serializer)
}

impl<'de> Deserialize<'de> for Event {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // 先读取 type，再只解析对应的具体事件，不能按变体顺序尝试匹配。
        // TitleEvent 的字段都有默认值，按顺序尝试会把消息等误读为标题并丢失内容。
        let value = Value::deserialize(deserializer)?;
        let event_type = value
            .get("type")
            .ok_or_else(|| serde::de::Error::missing_field("type"))?;
        let event_type: EventType =
            serde_json::from_value(event_type.clone()).map_err(serde::de::Error::custom)?;

        match event_type {
            EventType::Plan => serde_json::from_value(value).map(Self::Plan),
            EventType::Title => serde_json::from_value(value).map(Self::Title),
            EventType::Step => serde_json::from_value(value).map(Self::Step),
            EventType::Message => serde_json::from_value(value).map(Self::Message),
            EventType::Tool => serde_json::from_value(value).map(Self::Tool),
            EventType::Wait => serde_json::from_value(value).map(Self::Wait),
            EventType::Error => serde_json::from_value(value).map(Self::Error),
            EventType::Done => serde_json::from_value(value).map(Self::Done),
            EventType::Empty => {
                return Err(serde::de::Error::custom(
                    "application event type must not be empty",
                ));
            }
        }
        .map_err(serde::de::Error::custom)
    }
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
        A2aToolContent, BaseEvent, BrowserToolContent, DoneEvent, ErrorEvent, Event, EventType,
        FileToolContent, McpToolContent, MessageEvent, MessageRole, PlanEvent, PlanEventStatus,
        SearchToolContent, ShellToolContent, StepEvent, StepEventStatus, TitleEvent, ToolContent,
        ToolEvent, ToolEventStatus, WaitEvent,
    };
    use crate::domain::models::{File, Plan, SearchResultItem, Step, ToolResult};
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
    fn all_event_variants_round_trip_without_losing_payload() {
        let file = File {
            filename: "学习笔记.md".to_string(),
            filepath: "/workspace/学习笔记.md".to_string(),
            size: 128,
            ..File::default()
        };
        let step = Step {
            result: Some("读取完成".to_string()),
            attachments: vec![file.id.clone()],
            ..Step::new("读取课程资料")
        };
        let events = [
            (
                "plan",
                Event::Plan(PlanEvent {
                    plan: Plan {
                        steps: vec![step.clone()],
                        ..Plan::new("学习上下文工程", "整理设计思路")
                    },
                    status: PlanEventStatus::Updated,
                    ..PlanEvent::default()
                }),
            ),
            (
                "title",
                Event::Title(TitleEvent {
                    title: "课程学习笔记".to_string(),
                    ..TitleEvent::default()
                }),
            ),
            (
                "step",
                Event::Step(StepEvent {
                    step,
                    status: StepEventStatus::Completed,
                    ..StepEvent::default()
                }),
            ),
            (
                "message",
                Event::Message(MessageEvent {
                    role: MessageRole::User,
                    message: "请整理这份资料".to_string(),
                    attachments: vec![file],
                    ..MessageEvent::default()
                }),
            ),
            (
                "tool",
                Event::Tool(ToolEvent {
                    tool_call_id: "call-1".to_string(),
                    tool_name: "filesystem".to_string(),
                    tool_content: Some(ToolContent::Mcp(McpToolContent {
                        result: json!({"contents": "课程内容"}),
                    })),
                    function_name: "read_file".to_string(),
                    function_args: serde_json::from_value(json!({
                        "path": "/workspace/学习笔记.md",
                        "options": {"encoding": "utf-8"}
                    }))
                    .unwrap(),
                    function_result: Some(ToolResult {
                        success: true,
                        message: Some("读取成功".to_string()),
                        data: Some(json!({"contents": "课程内容", "bytes": 128})),
                    }),
                    status: ToolEventStatus::Called,
                    ..ToolEvent::default()
                }),
            ),
            ("wait", Event::Wait(WaitEvent::default())),
            (
                "error",
                Event::Error(ErrorEvent {
                    error: "文件读取失败".to_string(),
                    ..ErrorEvent::default()
                }),
            ),
            ("done", Event::Done(DoneEvent::default())),
        ];

        for (expected_type, event) in events {
            // 验证数据库 JSON 往返同时保留具体变体、业务内容、id 和创建时间。
            let value = serde_json::to_value(&event).unwrap();
            assert_eq!(value["type"], expected_type);
            assert!(value.get("base").is_none());
            assert!(value.get("id").is_some());
            assert!(value.get("created_at").is_some());
            let restored: Event = serde_json::from_value(value).unwrap();
            assert_eq!(restored, event, "{expected_type} event lost data");

            let text = serde_json::to_string(&event).unwrap();
            let restored: Event = serde_json::from_str(&text).unwrap();
            assert_eq!(restored, event, "{expected_type} event lost data");

            // 队列 id 替换初始 id 时，保留原有类型、时间和业务内容。
            let mut expected = serde_json::to_value(&event).unwrap();
            expected["id"] = json!("1750000000000-0");
            let mut event = event;
            event.set_id("1750000000000-0");
            assert_eq!(serde_json::to_value(event).unwrap(), expected);
        }
    }

    #[test]
    fn event_rejects_unknown_type() {
        let error = serde_json::from_value::<Event>(json!({"type": "future_event"}))
            .unwrap_err()
            .to_string();

        assert!(error.contains("unknown variant"), "{error}");
        assert!(error.contains("future_event"), "{error}");
    }

    #[test]
    fn event_requires_type_even_when_other_fields_match() {
        let error = serde_json::from_value::<Event>(json!({"title": "缺少类型"}))
            .unwrap_err()
            .to_string();

        assert!(error.contains("missing field `type`"), "{error}");
    }

    #[test]
    fn event_rejects_empty_base_event_type() {
        let error = serde_json::from_value::<Event>(json!({"type": ""}))
            .unwrap_err()
            .to_string();

        assert!(error.contains("type must not be empty"), "{error}");
    }

    #[test]
    fn malformed_tool_event_does_not_fall_back_to_title() {
        let error = serde_json::from_value::<Event>(json!({
            "type": "tool",
            "title": "不能作为标题事件读取"
        }))
        .unwrap_err()
        .to_string();

        assert!(error.contains("tool_call_id"), "{error}");
    }

    #[test]
    fn event_serialization_rejects_mismatched_variant_and_type() {
        let event = Event::Message(MessageEvent {
            base: BaseEvent {
                event_type: EventType::Title,
                ..BaseEvent::default()
            },
            message: "消息不能持久化为标题".to_string(),
            ..MessageEvent::default()
        });
        let error = serde_json::to_value(event).unwrap_err().to_string();

        assert!(error.contains("event type mismatch"), "{error}");
        assert!(error.contains("expected Message, found Title"), "{error}");
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
    fn tool_content_preserves_json_shape_and_event_round_trip() {
        let cases = [
            (
                ToolContent::Browser(BrowserToolContent {
                    screenshot: "snapshot.png".to_string(),
                }),
                json!({"screenshot": "snapshot.png"}),
            ),
            (
                ToolContent::Search(SearchToolContent {
                    results: vec![SearchResultItem {
                        url: "https://example.com".to_string(),
                        title: "课程资料".to_string(),
                        snippet: "工具内容设计".to_string(),
                    }],
                }),
                json!({"results": [{
                    "url": "https://example.com",
                    "title": "课程资料",
                    "snippet": "工具内容设计"
                }]}),
            ),
            (
                ToolContent::Shell(ShellToolContent {
                    console: json!([{"output": "执行完成", "exit_code": 0}]),
                }),
                json!({"console": [{"output": "执行完成", "exit_code": 0}]}),
            ),
            (
                ToolContent::File(FileToolContent {
                    content: "# 学习笔记\n工具内容\n".to_string(),
                }),
                json!({"content": "# 学习笔记\n工具内容\n"}),
            ),
            (
                ToolContent::Mcp(McpToolContent {
                    result: json!({"answer": 42}),
                }),
                json!({"result": {"answer": 42}}),
            ),
            (
                ToolContent::A2a(A2aToolContent {
                    a2a_result: json!({"artifacts": [{"text": "分析完成"}]}),
                }),
                json!({"a2a_result": {"artifacts": [{"text": "分析完成"}]}}),
            ),
        ];

        for (content, expected) in cases {
            // 保持各工具原有的 JSON 字段，并确保反序列化选择正确的枚举变体。
            assert_eq!(serde_json::to_value(&content).unwrap(), expected);
            let restored: ToolContent = serde_json::from_value(expected.clone()).unwrap();
            assert_eq!(restored, content);

            // 工具内容随事件保存和读取时，嵌套数据与具体类型都应完整保留。
            let event = Event::Tool(ToolEvent {
                tool_content: Some(content),
                ..ToolEvent::default()
            });
            let value = serde_json::to_value(&event).unwrap();
            assert_eq!(value["tool_content"], expected);
            assert_eq!(serde_json::from_value::<Event>(value).unwrap(), event);
            let text = serde_json::to_string(&event).unwrap();
            assert_eq!(serde_json::from_str::<Event>(&text).unwrap(), event);
        }
    }

    #[test]
    fn tool_content_preserves_arbitrary_json_results() {
        for payload in [
            Value::Null,
            json!(true),
            json!(42),
            json!("执行完成"),
            json!(["日志", null, 1]),
            json!({"nested": {"status": "done"}}),
        ] {
            let cases = [
                (
                    "console",
                    ToolContent::Shell(ShellToolContent {
                        console: payload.clone(),
                    }),
                ),
                (
                    "result",
                    ToolContent::Mcp(McpToolContent {
                        result: payload.clone(),
                    }),
                ),
                (
                    "a2a_result",
                    ToolContent::A2a(A2aToolContent {
                        a2a_result: payload.clone(),
                    }),
                ),
            ];
            for (field, expected) in cases {
                let value = json!({(field): payload});
                assert_eq!(serde_json::to_value(&expected).unwrap(), value);
                assert_eq!(
                    serde_json::from_value::<ToolContent>(value).unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    fn tool_content_requires_its_payload_field_and_type() {
        for invalid in [
            json!({}),
            json!({"screenshot": null}),
            json!({"results": null}),
            json!({"results": {}}),
            json!({"content": 123}),
        ] {
            assert!(
                serde_json::from_value::<ToolContent>(invalid.clone()).is_err(),
                "{invalid}"
            );
        }
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
