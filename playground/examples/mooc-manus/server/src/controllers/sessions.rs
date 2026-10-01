use axum::{
    http::StatusCode,
    response::{
        sse::{Event as SseEvent, Sse},
        IntoResponse,
    },
};
use chrono::{DateTime, Utc};
use futures::StreamExt;
use loco_rs::prelude::*;

use crate::{
    application::{error::AppError, services::session_service::SessionNotFound},
    interfaces::service_dependencies::{get_agent_service, get_session_service},
    openapi::{openapi, routes},
    views::sessions::{
        ChatRequest, CreateSessionResponse, EmptySessionData, ListSessionResponse, SessionResponse,
    },
};

/// 创建一个空白的新任务会话。
#[utoipa::path(
    post,
    path = "/api/sessions",
    tag = "会话模块",
    summary = "创建新任务会话",
    description = "创建一个空白的新任务会话，返回后续操作使用的 session_id。",
    responses(
        (status = 200, description = "创建任务会话成功", body = SessionResponse<CreateSessionResponse>),
        (status = 500, description = "会话创建失败")
    )
)]
#[debug_handler]
pub async fn create_session(State(ctx): State<AppContext>) -> Result<Response> {
    let session = get_session_service(&ctx)
        .create_session()
        .await
        .map_err(|error| map_session_error(error, "session.create_failed"))?;
    format::json(SessionResponse::success(
        "创建任务会话成功",
        CreateSessionResponse {
            session_id: session.id,
        },
    ))
}

/// 获取项目中所有任务会话的基础信息列表。
#[utoipa::path(
    get,
    path = "/api/sessions",
    tag = "会话模块",
    summary = "获取会话列表基础信息",
    description = "获取所有任务会话的标题、最新消息、时间、状态和未读数。",
    responses(
        (status = 200, description = "获取任务会话列表成功", body = SessionResponse<ListSessionResponse>),
        (status = 500, description = "会话列表读取失败")
    )
)]
#[debug_handler]
pub async fn get_all_sessions(State(ctx): State<AppContext>) -> Result<Response> {
    let sessions = get_session_service(&ctx)
        .get_all_sessions()
        .await
        .map_err(|error| map_session_error(error, "session.list_failed"))?;
    // 列表只转换基础信息，完整事件由后续会话详情和事件流接口提供。
    format::json(SessionResponse::success(
        "获取任务会话列表成功",
        ListSessionResponse {
            sessions: sessions.into_iter().map(Into::into).collect(),
        },
    ))
}

/// 根据传递的会话 id 清空未读消息数。
#[utoipa::path(
    post,
    path = "/api/sessions/{session_id}/clear-unread-message-count",
    tag = "会话模块",
    summary = "清除指定任务会话未读消息数",
    description = "将指定任务会话的未读消息数设为零。",
    params(("session_id" = String, Path, description = "会话业务 UUID")),
    responses(
        (status = 200, description = "清除未读消息数成功", body = SessionResponse<Option<EmptySessionData>>),
        (status = 400, description = "会话 UUID 无效"),
        (status = 500, description = "未读消息数更新失败")
    )
)]
#[debug_handler]
pub async fn clear_unread_message_count(
    State(ctx): State<AppContext>,
    Path(session_id): Path<String>,
) -> Result<Response> {
    validate_session_id(&session_id)?;
    get_session_service(&ctx)
        .clear_unread_message_count(&session_id)
        .await
        .map_err(|error| map_session_error(error, "session.clear_unread_failed"))?;
    format::json(SessionResponse::success(
        "清除未读消息数成功",
        None::<EmptySessionData>,
    ))
}

/// 根据传递的会话 id 删除指定任务会话。
#[utoipa::path(
    post,
    path = "/api/sessions/{session_id}/delete",
    tag = "会话模块",
    summary = "删除指定任务会话",
    description = "检查会话存在后删除会话记录。",
    params(("session_id" = String, Path, description = "会话业务 UUID")),
    responses(
        (status = 200, description = "删除任务会话成功", body = SessionResponse<Option<EmptySessionData>>),
        (status = 400, description = "会话 UUID 无效"),
        (status = 404, description = "会话不存在"),
        (status = 500, description = "会话删除失败")
    )
)]
#[debug_handler]
pub async fn delete_session(
    State(ctx): State<AppContext>,
    Path(session_id): Path<String>,
) -> Result<Response> {
    validate_session_id(&session_id)?;
    get_session_service(&ctx)
        .delete_session(&session_id)
        .await
        .map_err(|error| map_session_error(error, "session.delete_failed"))?;
    format::json(SessionResponse::success(
        "删除任务会话成功",
        None::<EmptySessionData>,
    ))
}

/// 根据会话 id 和聊天请求数据，向指定会话发起聊天请求。
#[utoipa::path(
    post,
    path = "/api/sessions/{session_id}/chat",
    tag = "会话模块",
    summary = "向指定任务会话发起聊天请求",
    description = "发送消息或订阅已有任务，逐条返回领域事件的 SSE 数据。",
    params(("session_id" = String, Path, description = "会话业务 UUID")),
    request_body = ChatRequest,
    responses(
        (status = 200, description = "聊天事件流", body = String, content_type = "text/event-stream"),
        (status = 400, description = "会话 UUID、时间戳或请求 JSON 无效"),
        (status = 422, description = "聊天请求字段类型无效"),
        (status = 500, description = "Agent 服务初始化失败")
    )
)]
#[debug_handler]
pub async fn chat(
    State(ctx): State<AppContext>,
    Path(session_id): Path<String>,
    Json(request): Json<ChatRequest>,
) -> Result<Response> {
    validate_session_id(&session_id)?;
    // 请求使用 Unix 秒，进入应用服务前转换为明确的 UTC 时间；0 表示纪元起点。
    let timestamp = request
        .timestamp
        .map(|seconds| {
            DateTime::<Utc>::from_timestamp(seconds, 0).ok_or_else(|| {
                AppError::bad_request("session.invalid_timestamp", "时间戳超出支持范围")
            })
        })
        .transpose()?;
    let service = get_agent_service(&ctx)
        .await
        .map_err(|error| AppError::internal("session.agent_init_failed", format!("{error:#}")))?;

    // 1.调用 Agent 服务发起聊天，将请求 event_id 传给 latest_event_id 参数。
    let events = service.chat(
        session_id,
        request.message,
        request.attachments,
        request.event_id,
        timestamp,
    );
    // 定义事件生成器，配合 Sse 生成流式响应数据。
    // 2.将 Agent 领域事件转换为 SSE 数据。
    // TODO: 后续与获取所有流式数据的接口统一响应结构。
    let stream = events.map(encode_sse_event);
    Ok(Sse::new(stream).into_response())
}

fn encode_sse_event(
    event: crate::domain::models::Event,
) -> std::result::Result<SseEvent, axum::Error> {
    let data = serde_json::to_value(event).map_err(axum::Error::new)?;
    let event_type = data
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| <serde_json::Error as serde::ser::Error>::custom("领域事件缺少 type 字段"))
        .map_err(axum::Error::new)?;
    // 事件 id 保留在 JSON 中，本课沿用 event + data 两个 SSE 字段。
    SseEvent::default().event(event_type).json_data(data)
}

fn validate_session_id(session_id: &str) -> std::result::Result<(), AppError> {
    uuid::Uuid::parse_str(session_id)
        .map(|_| ())
        .map_err(|_| AppError::bad_request("session.invalid_id", "会话 id 必须是有效的 UUID"))
}

fn map_session_error(error: anyhow::Error, code: &'static str) -> AppError {
    if error.is::<SessionNotFound>() {
        AppError::business(
            StatusCode::NOT_FOUND,
            "session.not_found",
            error.to_string(),
            None,
        )
    } else {
        AppError::internal(code, format!("{error:#}"))
    }
}

pub fn routes() -> Routes {
    Routes::new()
        .prefix("/api/sessions")
        .add(
            "/",
            openapi(
                post(create_session).get(get_all_sessions),
                routes!(create_session, get_all_sessions),
            ),
        )
        .add(
            "/{session_id}/clear-unread-message-count",
            openapi(
                post(clear_unread_message_count),
                routes!(clear_unread_message_count),
            ),
        )
        .add(
            "/{session_id}/delete",
            openapi(post(delete_session), routes!(delete_session)),
        )
        .add("/{session_id}/chat", openapi(post(chat), routes!(chat)))
}
