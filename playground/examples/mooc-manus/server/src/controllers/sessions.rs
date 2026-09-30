use axum::http::StatusCode;
use loco_rs::prelude::*;

use crate::{
    application::{error::AppError, services::session_service::SessionNotFound},
    interfaces::service_dependencies::get_session_service,
    openapi::{openapi, routes},
    views::sessions::{
        CreateSessionResponse, EmptySessionData, ListSessionResponse, SessionResponse,
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
}
