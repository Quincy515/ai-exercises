use std::time::Duration;

use axum::{
    extract::ws::{CloseFrame, Message as WebSocketMessage, WebSocket, WebSocketUpgrade},
    http::StatusCode,
    response::{
        sse::{Event as SseEvent, Sse},
        IntoResponse,
    },
};
use chrono::{DateTime, Utc};
use futures::{stream, SinkExt, StreamExt};
use loco_rs::prelude::*;
use tokio_tungstenite::{connect_async, tungstenite::Message as SandboxMessage};

use crate::{
    application::{
        error::AppError,
        services::session_service::{SessionNotFound, SessionSandboxError},
    },
    interfaces::service_dependencies::{get_agent_service, get_session_service},
    openapi::{openapi, routes},
    views::{
        events::AgentSseEvent,
        sessions::{
            ChatRequest, CreateSessionResponse, EmptySessionData, FileReadRequest,
            FileReadResponse, GetSessionFilesResponse, GetSessionResponse, ListSessionResponse,
            SessionResponse, ShellReadRequest, ShellReadResponse,
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
    let session = get_session_service(&ctx)?
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
    let service = get_session_service(&ctx)?;
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
    let sessions = get_session_service(&ctx)?
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
    get_session_service(&ctx)?
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
    get_session_service(&ctx)?
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
    let session = get_session_service(&ctx)?
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

/// 根据传递的指定会话 id 停止对应任务会话。
#[utoipa::path(
    post,
    path = "/api/sessions/{session_id}/stop",
    tag = "会话模块",
    summary = "停止指定任务会话",
    description = "查找并取消已有任务，将指定会话标记为已完成。",
    params(("session_id" = String, Path, description = "会话业务 UUID")),
    responses(
        (status = 200, description = "停止任务会话成功", body = SessionResponse<Option<EmptySessionData>>),
        (status = 400, description = "会话 UUID 无效"),
        (status = 500, description = "服务初始化失败、会话不存在或停止失败")
    )
)]
#[debug_handler]
pub async fn stop_session(
    State(ctx): State<AppContext>,
    Path(session_id): Path<String>,
) -> Result<Response> {
    validate_session_id(&session_id)?;
    get_agent_service(&ctx)
        .await
        .map_err(|error| AppError::internal("session.agent_init_failed", format!("{error:#}")))?
        .stop_session(&session_id)
        .await
        .map_err(|error| map_session_error(error, "session.stop_failed"))?;
    format::json(SessionResponse::success(
        "停止任务会话成功",
        None::<EmptySessionData>,
    ))
}

/// 获取指定任务会话文件列表信息。
#[utoipa::path(
    get,
    path = "/api/sessions/{session_id}/files",
    tag = "会话模块",
    summary = "获取指定任务会话文件列表信息",
    description = "返回当前会话中人类上传与智能体生成的全部文件信息。",
    params(("session_id" = String, Path, description = "会话业务 UUID")),
    responses(
        (status = 200, description = "获取会话文件列表成功", body = SessionResponse<GetSessionFilesResponse>),
        (status = 400, description = "会话 UUID 无效"),
        (status = 500, description = "会话不存在或文件列表读取失败")
    )
)]
#[debug_handler]
pub async fn get_session_files(
    State(ctx): State<AppContext>,
    Path(session_id): Path<String>,
) -> Result<Response> {
    validate_session_id(&session_id)?;
    let files = get_session_service(&ctx)?
        .get_session_files(&session_id)
        .await
        .map_err(|error| map_session_error(error, "session.files_failed"))?;
    format::json(SessionResponse::success(
        "获取会话文件列表成功",
        GetSessionFilesResponse {
            files: files.into_iter().map(Into::into).collect(),
        },
    ))
}

/// 根据传递的会话 id + 文件路径查看沙箱中文件的内容信息。
#[utoipa::path(
    post,
    path = "/api/sessions/{session_id}/file",
    tag = "会话模块",
    summary = "查看会话沙箱中指定文件的内容",
    description = "根据传递的会话 id 与文件路径查看沙箱中文件的内容信息。",
    params(("session_id" = String, Path, description = "会话业务 UUID")),
    request_body = FileReadRequest,
    responses(
        (status = 200, description = "获取会话文件内容成功", body = SessionResponse<FileReadResponse>),
        (status = 400, description = "会话 UUID 无效或 JSON 格式错误"),
        (status = 404, description = "当前会话无沙箱或沙箱已销毁"),
        (status = 422, description = "请求字段缺失或类型错误"),
        (status = 500, description = "会话不存在、沙箱读取失败或响应格式错误")
    )
)]
#[debug_handler]
pub async fn read_file(
    State(ctx): State<AppContext>,
    Path(session_id): Path<String>,
    Json(request): Json<FileReadRequest>,
) -> Result<Response> {
    validate_session_id(&session_id)?;
    let content = get_session_service(&ctx)?
        .read_file(&session_id, &request.filepath)
        .await
        .map_err(|error| map_session_error(error, "session.file_read_failed"))?;
    // 当前沙箱原样回显请求路径，适配器已从结果中提取文件文本。
    format::json(SessionResponse::success(
        "获取会话文件内容成功",
        FileReadResponse {
            filepath: request.filepath,
            content,
        },
    ))
}

/// 查看会话的 Shell 内容输出。
#[utoipa::path(
    post,
    path = "/api/sessions/{session_id}/shell",
    tag = "会话模块",
    summary = "查看会话的 Shell 内容输出",
    description = "传递指定任务会话 id 与 Shell 会话标识，查看输出和控制台记录。",
    params(("session_id" = String, Path, description = "任务会话业务 UUID，与请求体中的 Shell 会话 id 分开")),
    request_body = ShellReadRequest,
    responses(
        (status = 200, description = "获取 Shell 内容输出结果成功", body = SessionResponse<ShellReadResponse>),
        (status = 400, description = "会话 UUID 无效或 JSON 格式错误"),
        (status = 404, description = "当前会话无沙箱或沙箱已销毁"),
        (status = 422, description = "请求字段缺失或类型错误"),
        (status = 500, description = "会话不存在、沙箱读取失败或响应格式错误")
    )
)]
#[debug_handler]
pub async fn read_shell_output(
    State(ctx): State<AppContext>,
    Path(session_id): Path<String>,
    Json(request): Json<ShellReadRequest>,
) -> Result<Response> {
    validate_session_id(&session_id)?;
    let data = get_session_service(&ctx)?
        .read_shell_output(&session_id, &request.session_id)
        .await
        .map_err(|error| map_session_error(error, "session.shell_read_failed"))?;
    // 只保留约定的响应字段；缺失控制台记录时使用空列表。
    let response: ShellReadResponse = serde_json::from_str(&data)
        .map_err(|error| AppError::internal("session.shell_response_invalid", error.to_string()))?;
    format::json(SessionResponse::success(
        "获取Shell内容输出结果成功",
        response,
    ))
}

/// VNC WebSocket 端点，建立沙箱连接并双向转发数据。
pub async fn vnc_websocket(
    State(ctx): State<AppContext>,
    Path(session_id): Path<String>,
    websocket: WebSocketUpgrade,
) -> Result<Response> {
    let service = get_session_service(&ctx)?;
    tracing::info!(session_id, "为会话开启 WebSocket 连接");
    // 1.从客户端 noVNC 接收子协议；2.binary 优先，base64 次选。
    // Axum 按服务端提供的顺序选择客户端支持的协议。
    // 3.使用对应协议接受 WebSocket 连接，随后查询会话及沙箱。
    Ok(websocket
        .protocols(["binary", "base64"])
        .on_upgrade(move |mut socket| async move {
            let result: anyhow::Result<()> = async {
                // 4.获取对应会话的 VNC 链接。
                let sandbox_vnc_url = service.get_vnc_url(&session_id).await?;
                tracing::info!(session_id, sandbox_vnc_url, "连接 WebSocket VNC");
                forward_vnc(&mut socket, &sandbox_vnc_url).await
            }
            .await;
            if let Err(error) = result {
                // 连接沙箱失败或其他异常：记录日志并使用 1011 关闭 WebSocket。
                tracing::error!(session_id, error = %error, "WebSocket 异常");
                close_vnc_with_error(&mut socket, format!("WebSocket异常: {error:#}")).await;
            }
        }))
}

async fn forward_vnc(websocket: &mut WebSocket, sandbox_vnc_url: &str) -> anyhow::Result<()> {
    use anyhow::Context;

    // 5.连接到 VNC；沿用客户端连接的十秒握手等待上限。
    let (sandbox_ws, _) =
        tokio::time::timeout(Duration::from_secs(10), connect_async(sandbox_vnc_url))
            .await
            .context("连接沙箱环境超时")?
            .context("连接沙箱环境失败")?;
    let (mut web_sender, mut web_receiver) = websocket.split();
    let (mut sandbox_sender, mut sandbox_receiver) = sandbox_ws.split();

    // 6.创建两个异步 future 完成数据的双向转发。
    let forward_to_sandbox = async {
        while let Some(message) = web_receiver.next().await {
            match message? {
                // 接收来自客户端的数据，原样发送给沙箱。
                WebSocketMessage::Binary(data) => {
                    sandbox_sender.send(SandboxMessage::Binary(data)).await?
                }
                WebSocketMessage::Close(_) => break,
                WebSocketMessage::Ping(_) | WebSocketMessage::Pong(_) => {}
                WebSocketMessage::Text(_) => anyhow::bail!("VNC 客户端需要发送二进制消息"),
            }
        }
        Ok::<_, anyhow::Error>(())
    };
    let forward_from_sandbox = async {
        while let Some(message) = sandbox_receiver.next().await {
            match message? {
                // 接收来自沙箱的数据并转发给客户端。
                SandboxMessage::Binary(data) => {
                    web_sender.send(WebSocketMessage::Binary(data)).await?
                }
                SandboxMessage::Close(_) => break,
                SandboxMessage::Ping(_) | SandboxMessage::Pong(_) | SandboxMessage::Frame(_) => {}
                SandboxMessage::Text(_) => anyhow::bail!("VNC 沙箱需要发送二进制消息"),
            }
        }
        Ok::<_, anyhow::Error>(())
    };

    // 7.并行运行两个方向；8.等待任意方向结束，表示连接已中断。
    // select! 返回时丢弃另一方向的 future，对应取消剩余转发任务。
    let (direction, result) = tokio::select! {
        result = forward_to_sandbox => ("Web->VNC", result),
        result = forward_from_sandbox => ("VNC->Web", result),
    };
    if let Err(error) = result {
        tracing::error!(direction, error = %error, "VNC 转发出错");
    }
    tracing::info!(direction, "WebSocket 连接已关闭");

    // 9.关闭两端连接；限时发送关闭帧，结束后由所有权释放连接资源。
    let _ = tokio::time::timeout(Duration::from_secs(1), async {
        tokio::join!(web_sender.close(), sandbox_sender.close())
    })
    .await;
    Ok(())
}

async fn close_vnc_with_error(websocket: &mut WebSocket, mut reason: String) {
    // WebSocket 关闭帧最多携带 123 字节原因，截断时保留完整 UTF-8 字符。
    let mut end = reason.len().min(123);
    while !reason.is_char_boundary(end) {
        end -= 1;
    }
    reason.truncate(end);
    let _ = tokio::time::timeout(
        Duration::from_secs(1),
        websocket.send(WebSocketMessage::Close(Some(CloseFrame {
            code: 1011,
            reason: reason.into(),
        }))),
    )
    .await;
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
    if let Some(sandbox_error) = error.downcast_ref::<SessionSandboxError>() {
        let (status, code) = match sandbox_error {
            SessionSandboxError::Unassigned | SessionSandboxError::Unavailable => {
                (StatusCode::NOT_FOUND, "session.sandbox_not_found")
            }
            SessionSandboxError::RequestFailed(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "session.sandbox_request_failed",
            ),
        };
        AppError::business(status, code, error.to_string(), None)
    } else if error.is::<SessionNotFound>() {
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
        .add(
            "/{session_id}/stop",
            openapi(post(stop_session), routes!(stop_session)),
        )
        .add(
            "/{session_id}/files",
            openapi(get(get_session_files), routes!(get_session_files)),
        )
        .add(
            "/{session_id}/file",
            openapi(post(read_file), routes!(read_file)),
        )
        .add(
            "/{session_id}/shell",
            openapi(post(read_shell_output), routes!(read_shell_output)),
        )
        // WebSocket 升级使用 GET，保持为独立于 REST OpenAPI 的长连接入口。
        .add("/{session_id}/vnc", get(vnc_websocket))
}

#[cfg(test)]
#[path = "sessions_vnc_tests.rs"]
mod vnc_tests;

#[cfg(test)]
mod tests {
    use super::{map_session_error, SessionSandboxError, StatusCode};
    use loco_rs::Error;

    #[test]
    fn sandbox_business_errors_keep_their_status_and_message() {
        for (error, status, code, message) in [
            (
                SessionSandboxError::Unassigned,
                StatusCode::NOT_FOUND,
                "session.sandbox_not_found",
                "当前会话无沙箱环境",
            ),
            (
                SessionSandboxError::Unavailable,
                StatusCode::NOT_FOUND,
                "session.sandbox_not_found",
                "当前会话沙箱不存在或已销毁",
            ),
            (
                SessionSandboxError::RequestFailed("文件读取失败".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "session.sandbox_request_failed",
                "文件读取失败",
            ),
        ] {
            let error: Error = map_session_error(error.into(), "session.read_failed").into();
            let Error::CustomError(actual_status, detail) = error else {
                panic!("预期业务错误响应");
            };
            assert_eq!(actual_status, status);
            assert_eq!(detail.error.as_deref(), Some(code));
            assert_eq!(detail.description.as_deref(), Some(message));
        }
    }
}
