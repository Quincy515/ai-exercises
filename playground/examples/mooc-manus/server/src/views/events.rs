//! 会话流式响应结构：在接口边界投影领域事件，供 Web 与 Electron 共用。
//! 统一转换领域事件与事件列表，控制器将响应数据编码成 SSE 帧。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use utoipa::{
    openapi::{
        schema::{AnyOfBuilder, ObjectBuilder, Schema, Type},
        RefOr,
    },
    PartialSchema, ToSchema,
};

use crate::domain::models::{
    BaseEvent, ErrorEvent, Event, ExecutionStatus, MessageEvent, MessageRole, PlanEvent, Step,
    StepEvent, TitleEvent, ToolContent, ToolEvent, ToolEventStatus,
};

use super::files::FileInfoResponse;

/// 基础事件数据。
#[derive(Debug, Serialize, ToSchema)]
pub struct BaseEventData {
    /// 事件 id，对应消息队列的事件标识，与步骤和文件的业务 id 分开。
    pub event_id: Option<String>,
    /// 事件时间，响应中统一使用整数 Unix 秒。
    pub created_at: i64,
}

impl Default for BaseEventData {
    fn default() -> Self {
        Self {
            event_id: None,
            created_at: Utc::now().timestamp(),
        }
    }
}

impl From<&BaseEvent> for BaseEventData {
    /// 从领域模型中构建基础事件数据，在响应边界将时间转换为 Unix 秒。
    fn from(event: &BaseEvent) -> Self {
        Self {
            event_id: Some(event.id.clone()),
            created_at: event.created_at.timestamp(),
        }
    }
}

/// 通用事件数据，允许保留额外字段。
#[derive(Debug, Serialize, ToSchema)]
pub struct CommonEventData {
    #[serde(flatten)]
    pub base: BaseEventData,
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub extra: Map<String, Value>,
}

/// 通用流式事件，事件名和额外数据由调用方提供。
#[derive(Debug, Serialize, ToSchema)]
pub struct CommonSseEvent {
    /// 事件类型。
    pub event: String,
    /// 事件数据。
    pub data: CommonEventData,
}

impl TryFrom<&Event> for CommonSseEvent {
    type Error = serde_json::Error;

    /// 将领域事件转换成通用流式事件，保留公共字段之外的数据。
    fn try_from(event: &Event) -> Result<Self, Self::Error> {
        // 1.提取领域事件的公共字段，其余字段作为通用事件的额外数据。
        #[derive(Deserialize)]
        struct EventFields {
            id: Option<String>,
            created_at: DateTime<Utc>,
            #[serde(rename = "type")]
            event: String,
            #[serde(flatten)]
            extra: Map<String, Value>,
        }

        let fields: EventFields = serde_json::from_value(serde_json::to_value(event)?)?;

        // 2.构建 event + data 响应，事件类型只保留在外层。
        Ok(Self {
            event: fields.event,
            data: CommonEventData {
                base: BaseEventData {
                    event_id: fields.id,
                    created_at: fields.created_at.timestamp(),
                },
                extra: fields.extra,
            },
        })
    }
}

/// 消息事件数据。
#[derive(Debug, Default, Serialize, ToSchema)]
pub struct MessageEventData {
    #[serde(flatten)]
    pub base: BaseEventData,
    /// 消息角色，默认为 assistant。
    #[schema(value_type = String, example = "assistant")]
    pub role: MessageRole,
    /// 消息内容，默认为空字符串。
    pub message: String,
    /// 附件列表信息，复用文件接口的响应结构，默认为空列表。
    pub attachments: Vec<FileInfoResponse>,
}

impl From<MessageEvent> for MessageEventData {
    fn from(event: MessageEvent) -> Self {
        Self {
            base: (&event.base).into(),
            role: event.role,
            message: event.message,
            attachments: event.attachments.into_iter().map(Into::into).collect(),
        }
    }
}

/// 标题事件数据。
#[derive(Debug, Serialize, ToSchema)]
pub struct TitleEventData {
    #[serde(flatten)]
    pub base: BaseEventData,
    /// 会话标题。
    pub title: String,
}

impl From<TitleEvent> for TitleEventData {
    fn from(event: TitleEvent) -> Self {
        Self {
            base: (&event.base).into(),
            title: event.title,
        }
    }
}

/// 步骤事件数据，只保留 UI 展示需要的步骤信息。
#[derive(Debug, Serialize, ToSchema)]
pub struct StepEventData {
    #[serde(flatten)]
    pub base: BaseEventData,
    /// 步骤 id；独立于公共 event_id，供 UI 关联同一个步骤。
    pub id: String,
    /// 步骤执行状态：pending、running、completed、failed。
    #[schema(value_type = String, example = "running")]
    pub status: ExecutionStatus,
    /// 步骤描述。
    pub description: String,
}

impl StepEventData {
    fn from_step(step: Step, event: &BaseEvent) -> Self {
        Self {
            base: event.into(),
            id: step.id,
            status: step.status,
            description: step.description,
        }
    }
}

impl From<StepEvent> for StepEventData {
    fn from(event: StepEvent) -> Self {
        // 展开嵌套 step，使用步骤执行状态和步骤 id。
        Self::from_step(event.step, &event.base)
    }
}

/// 计划事件数据。
#[derive(Debug, Serialize, ToSchema)]
pub struct PlanEventData {
    #[serde(flatten)]
    pub base: BaseEventData,
    /// 计划中的步骤列表。
    pub steps: Vec<StepEventData>,
}

impl From<PlanEvent> for PlanEventData {
    fn from(event: PlanEvent) -> Self {
        // 1.保留计划事件的公共信息。
        let base = BaseEventData::from(&event.base);
        // 2.将计划中的各步骤投影成统一响应，沿用计划事件的 event_id 和时间。
        let steps = event
            .plan
            .steps
            .into_iter()
            .map(|step| StepEventData::from_step(step, &event.base))
            .collect();
        Self { base, steps }
    }
}

/// 工具事件数据。
#[derive(Debug, Serialize, ToSchema)]
pub struct ToolEventData {
    #[serde(flatten)]
    pub base: BaseEventData,
    /// 工具调用 id。
    pub tool_call_id: String,
    /// 工具箱名字。
    pub name: String,
    /// 工具状态：calling、called。
    #[schema(value_type = String, example = "calling")]
    pub status: ToolEventStatus,
    /// 工具名字。
    pub function: String,
    /// 工具参数。
    #[schema(value_type = Object)]
    pub args: Map<String, Value>,
    /// 工具展示内容，如截图、搜索条目、控制台记录或文件正文；尚无内容时为 null。
    #[schema(value_type = Option<Object>)]
    pub content: Option<ToolContent>,
}

impl From<ToolEvent> for ToolEventData {
    fn from(event: ToolEvent) -> Self {
        Self {
            base: (&event.base).into(),
            tool_call_id: event.tool_call_id,
            name: event.tool_name,
            status: event.status,
            function: event.function_name,
            args: event.function_args,
            content: event.tool_content,
        }
    }
}

/// 错误事件数据。
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorEventData {
    #[serde(flatten)]
    pub base: BaseEventData,
    /// 错误信息。
    pub error: String,
}

impl From<ErrorEvent> for ErrorEventData {
    fn from(event: ErrorEvent) -> Self {
        Self {
            base: (&event.base).into(),
            error: event.error,
        }
    }
}

/// Agent 流式事件类型集合，统一外层的事件类型与数据结构。
/// Rust 枚举把事件名与对应数据绑定，Serde 输出 {"event": "...", "data": {...}}。
/// utoipa 5.5 的 schema 派生仅支持容器级 untagged，聚合 schema 需单独描述真实信封。
/// 各具体响应数据类型继续通过 ToSchema 描述接口字段，联合类型在下面复用这些描述。
#[derive(Debug, Serialize)]
#[serde(tag = "event", content = "data", rename_all = "lowercase")]
pub enum AgentSseEvent {
    /// 流式消息事件数据响应结构。
    Message(MessageEventData),
    /// 标题流式事件。
    Title(TitleEventData),
    /// 步骤流式事件。
    Step(StepEventData),
    /// 计划流式事件。
    Plan(PlanEventData),
    /// 工具流式事件。
    Tool(ToolEventData),
    /// 停止流式事件，保留基础事件数据。
    Done(BaseEventData),
    /// 错误流式事件。
    Error(ErrorEventData),
    /// 等待人类输入流式事件，保留基础事件数据。
    Wait(BaseEventData),
    /// 通用事件直接沿用自身的 event + data 结构。
    #[serde(untagged)]
    Common(CommonSseEvent),
}

impl PartialSchema for AgentSseEvent {
    fn schema() -> RefOr<Schema> {
        // 描述实际的 event + data 信封，供会话详情的事件数组复用。
        // 通用事件可以与具体事件同时匹配，因此使用 anyOf。
        let mut schema = AnyOfBuilder::new();
        for (event, data) in [
            ("message", MessageEventData::schema()),
            ("title", TitleEventData::schema()),
            ("step", StepEventData::schema()),
            ("plan", PlanEventData::schema()),
            ("tool", ToolEventData::schema()),
            ("done", BaseEventData::schema()),
            ("error", ErrorEventData::schema()),
            ("wait", BaseEventData::schema()),
        ] {
            schema = schema.item(
                ObjectBuilder::new()
                    .property(
                        "event",
                        ObjectBuilder::new()
                            .schema_type(Type::String)
                            .enum_values(Some([event])),
                    )
                    .property("data", data)
                    .required("event")
                    .required("data"),
            );
        }
        schema.item(CommonSseEvent::schema()).into()
    }
}

impl ToSchema for AgentSseEvent {
    fn schemas(schemas: &mut Vec<(String, RefOr<Schema>)>) {
        // 注册各载荷引用到的嵌套类型，例如文件、步骤和基础事件字段。
        MessageEventData::schemas(schemas);
        TitleEventData::schemas(schemas);
        StepEventData::schemas(schemas);
        PlanEventData::schemas(schemas);
        ToolEventData::schemas(schemas);
        BaseEventData::schemas(schemas);
        ErrorEventData::schemas(schemas);
        CommonSseEvent::schemas(schemas);
    }
}

impl From<Event> for AgentSseEvent {
    /// 将领域事件转换为 Agent 流式事件模型。
    fn from(event: Event) -> Self {
        // 1.按 Rust 枚举变体选择具体响应类型，编译器检查事件是否全部覆盖。
        // 2.复用各数据类型的转换，将领域字段投影成适合客户端展示的数据。
        match event {
            Event::Message(event) => Self::Message(event.into()),
            Event::Title(event) => Self::Title(event.into()),
            Event::Step(event) => Self::Step(event.into()),
            Event::Plan(event) => Self::Plan(event.into()),
            Event::Tool(event) => Self::Tool(event.into()),
            Event::Done(event) => Self::Done((&event.base).into()),
            Event::Error(event) => Self::Error(event.into()),
            Event::Wait(event) => Self::Wait((&event.base).into()),
        }
    }
}

impl AgentSseEvent {
    /// 将领域事件模型列表转换为 SSE 流式事件列表，保持输入顺序。
    pub fn from_events(events: impl IntoIterator<Item = Event>) -> Vec<Self> {
        // 每个领域事件都有确定的响应类型，直接转换并收集即可。
        events.into_iter().map(Self::from).collect()
    }
}
