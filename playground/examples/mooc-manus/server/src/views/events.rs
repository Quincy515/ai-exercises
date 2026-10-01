//! 会话流式响应结构：在接口边界投影领域事件，供 Web 与 Electron 共用。
//! 本课定义数据结构和单类型转换；控制器的事件分发与 SSE 接入由后续课程完成。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use utoipa::ToSchema;

use crate::domain::models::{
    BaseEvent, ErrorEvent, Event, ExecutionStatus, MessageEvent, MessageRole, PlanEvent, Step,
    StepEvent, TitleEvent, ToolEvent, ToolEventStatus, ToolResult,
};

use super::files::FileInfoResponse;

/// 基础事件数据。
#[derive(Debug, Serialize, ToSchema)]
pub struct BaseEventData {
    /// 事件 id。
    pub id: Option<String>,
    /// 事件时间，响应中统一使用整数 Unix 秒。
    pub created_at: i64,
}

impl Default for BaseEventData {
    fn default() -> Self {
        Self {
            id: None,
            created_at: Utc::now().timestamp(),
        }
    }
}

impl From<&BaseEvent> for BaseEventData {
    /// 从领域模型中构建基础事件数据，在响应边界将时间转换为 Unix 秒。
    fn from(event: &BaseEvent) -> Self {
        Self {
            id: Some(event.id.clone()),
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
                    id: fields.id,
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
    /// 步骤 id；这里使用步骤的标识，避免与公共事件 id 重复。
    pub id: String,
    /// 事件时间，单位为 Unix 秒；计划内的步骤沿用计划事件时间。
    pub created_at: i64,
    /// 步骤执行状态：pending、running、completed、failed。
    #[schema(value_type = String, example = "running")]
    pub status: ExecutionStatus,
    /// 步骤描述。
    pub description: String,
}

impl StepEventData {
    fn from_step(step: Step, created_at: i64) -> Self {
        Self {
            id: step.id,
            created_at,
            status: step.status,
            description: step.description,
        }
    }
}

impl From<StepEvent> for StepEventData {
    fn from(event: StepEvent) -> Self {
        // 展开嵌套 step，使用步骤执行状态和步骤 id。
        Self::from_step(event.step, event.base.created_at.timestamp())
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
        // 2.将计划中的各步骤投影成统一的步骤响应。
        let steps = event
            .plan
            .steps
            .into_iter()
            .map(|step| StepEventData::from_step(step, base.created_at))
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
    /// 工具调用结果，保留完整的 success/message/data；尚无结果时为 null。
    #[schema(value_type = Option<Object>)]
    pub content: Option<ToolResult<Value>>,
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
            content: event.function_result,
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
/// utoipa 5.5 的 schema 派生仅支持容器级 untagged；聚合 schema 留待端点接入时定义。
/// 各具体响应数据类型继续通过 ToSchema 描述接口字段。
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
