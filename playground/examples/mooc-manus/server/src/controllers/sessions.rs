use std::time::Duration;

use axum::{
    http::StatusCode,
    response::{
        sse::{Event as SseEvent, Sse},
        IntoResponse,
    },
};
use chrono::{DateTime, Utc};
use futures::{stream, StreamExt};
use loco_rs::prelude::*;

use crate::{
    application::{error::AppError, services::session_service::SessionNotFound},
    interfaces::service_dependencies::{get_agent_service, get_session_service},
    openapi::{openapi, routes},
    views::{
        events::AgentSseEvent,
        sessions::{
            ChatRequest, CreateSessionResponse, EmptySessionData, GetSessionResponse,
            ListSessionResponse, SessionResponse,
        },
    },
};

/// 流式获取会话列表的睡眠间隔。
const SESSION_SLEEP_INTERVAL: Duration = Duration::from_secs(5);

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

/// 间隔指定时间流式获取所有会话基础信息列表。
#[utoipa::path(
    post,
    path = "/api/sessions/stream",
    tag = "会话模块",
    summary = "流式获取所有会话基础信息列表",
    description = "立即返回所有会话基础信息，随后每隔五秒重新查询并返回全量列表。",
    responses(
        (status = 200, description = "sessions 事件流，data 为会话基础信息列表", body = String, content_type = "text/event-stream")
    )
)]
#[debug_handler]
pub async fn stream_sessions(State(ctx): State<AppContext>) -> Result<Response> {
    let service = get_session_service(&ctx);
    // 定义事件生成器：首轮立即读取，后续在上次事件交付后等待五秒。
    // 请求断开时生成器被丢弃，数据库查询结束后再等待，不持有长事务。
    let events = stream::try_unfold((service, false), |(service, should_sleep)| async move {
        // 4.睡眠指定时间，避免高频响应；首次查询直接执行。
        if should_sleep {
            tokio::time::sleep(SESSION_SLEEP_INTERVAL).await;
        }
        // 1.获取所有会话列表。
        let sessions = service.get_all_sessions().await.map_err(axum::Error::new)?;
        // 2.循环遍历并组装基础信息。
        let data = ListSessionResponse {
            sessions: sessions.into_iter().map(Into::into).collect(),
        };
        // 3.将会话列表转换为流式事件数据并返回。
        let event = SseEvent::default().event("sessions").json_data(data)?;
        Ok::<_, axum::Error>(Some((event, (service, true))))
    });
    Ok(Sse::new(events).into_response())
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
    // 列表只转换基础信息，完整事件由会话详情和聊天事件流接口提供。
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
    // 2.将 Agent 领域事件转换为统一响应，再将事件名与数据分别写入 SSE 帧。
    let stream = events.map(encode_sse_event);
    Ok(Sse::new(stream).into_response())
}

/// 传递指定会话 id 获取该会话的对话详情。
#[utoipa::path(
    get,
    path = "/api/sessions/{session_id}",
    tag = "会话模块",
    summary = "获取指定会话详情信息",
    description = "根据会话 id 获取标题、状态和对话过程中产生的全部事件。",
    params(("session_id" = String, Path, description = "会话业务 UUID")),
    responses(
        (status = 200, description = "获取会话详情成功", body = SessionResponse<GetSessionResponse>),
        (status = 400, description = "会话 UUID 无效"),
        (status = 404, description = "会话不存在"),
        (status = 500, description = "会话详情读取失败")
    )
)]
#[debug_handler]
pub async fn get_session(
    State(ctx): State<AppContext>,
    Path(session_id): Path<String>,
) -> Result<Response> {
    validate_session_id(&session_id)?;
    let session = get_session_service(&ctx)
        .get_session(&session_id)
        .await
        .map_err(|error| map_session_error(error, "session.get_failed"))?
        .ok_or_else(|| {
            AppError::business(
                StatusCode::NOT_FOUND,
                "session.not_found",
                "该会话不存在，请核实后重试",
                None,
            )
        })?;
    format::json(SessionResponse::success(
        "获取会话详情成功",
        GetSessionResponse::from(session),
    ))
}

fn encode_sse_event(
    event: crate::domain::models::Event,
) -> std::result::Result<SseEvent, axum::Error> {
    let response = serde_json::to_value(AgentSseEvent::from(event)).map_err(axum::Error::new)?;
    let event_type = response
        .get("event")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| <serde_json::Error as serde::ser::Error>::custom("响应事件缺少 event 字段"))
        .map_err(axum::Error::new)?;
    let data = response
        .get("data")
        .ok_or_else(|| <serde_json::Error as serde::ser::Error>::custom("响应事件缺少 data 字段"))
        .map_err(axum::Error::new)?;
    // data 只包含响应载荷；event_id 保留在 JSON 中，SSE event 字段保存事件名称。
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
            "/stream",
            openapi(post(stream_sessions), routes!(stream_sessions)),
        )
        .add(
            "/{session_id}",
            openapi(get(get_session), routes!(get_session)),
        )
        .add(
            "/{session_id}/delete",
            openapi(post(delete_session), routes!(delete_session)),
        )
        .add("/{session_id}/chat", openapi(post(chat), routes!(chat)))
}
