use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};

use crate::{
    exceptions::{ApiResponse, AppException},
    models::{ProcessInfo, SupervisorActionResult, SupervisorTimeout},
    services::DEFAULT_TIMEOUT_EXTENSION_MINUTES,
    views::TimeoutRequest,
};

use super::service_dependencies::AppState;

pub struct SupervisorController;

#[utoipa::path(
    get,
    context_path = "/api",
    path = "/supervisor/status",
    tag = "Supervisor模块",
    description = "获取沙箱中所有进程服务的状态信息",
    responses(
        (status = 200, description = "获取沙箱进程服务成功", body = ApiResponse<Vec<ProcessInfo>>),
        (status = 500, description = "获取沙箱进程服务失败", body = ApiResponse),
    ),
)]
async fn get_status(
    State(state): State<AppState>,
) -> Result<ApiResponse<Vec<ProcessInfo>>, AppException> {
    let result = state.supervisor_service.get_all_processes().await?;

    Ok(ApiResponse::success(Some(result), "获取沙箱进程服务成功"))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/supervisor/stop-all-processes",
    tag = "Supervisor模块",
    description = "停止所有supervisor进程服务",
    responses(
        (status = 200, description = "停止Supervisor所有进程服务成功", body = ApiResponse<SupervisorActionResult>),
        (status = 500, description = "停止Supervisor所有进程服务失败", body = ApiResponse),
    ),
)]
async fn stop_all_processes(
    State(state): State<AppState>,
) -> Result<ApiResponse<SupervisorActionResult>, AppException> {
    let result = state.supervisor_service.stop_all_processes().await?;

    Ok(ApiResponse::success(
        Some(result),
        "停止Supervisor所有进程服务成功",
    ))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/supervisor/shutdown",
    tag = "Supervisor模块",
    description = "关闭supervisor服务本身",
    responses(
        (status = 200, description = "Supervisor服务关闭成功", body = ApiResponse<SupervisorActionResult>),
        (status = 500, description = "Supervisor服务关闭失败", body = ApiResponse),
    ),
)]
async fn shutdown(
    State(state): State<AppState>,
) -> Result<ApiResponse<SupervisorActionResult>, AppException> {
    let result = state.supervisor_service.shutdown().await?;

    Ok(ApiResponse::success(Some(result), "Supervisor服务关闭成功"))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/supervisor/restart",
    tag = "Supervisor模块",
    description = "重启supervisor管理的所有子进程",
    responses(
        (status = 202, description = "Supervisor重启任务已提交", body = ApiResponse<SupervisorActionResult>),
        (status = 400, description = "Supervisor重启任务正在执行", body = ApiResponse),
        (status = 500, description = "重启Supervisor所有进程服务失败", body = ApiResponse),
    ),
)]
async fn restart(
    State(state): State<AppState>,
) -> Result<(StatusCode, ApiResponse<SupervisorActionResult>), AppException> {
    let result = state.supervisor_service.schedule_restart()?;

    Ok((
        StatusCode::ACCEPTED,
        ApiResponse::accepted(Some(result), "Supervisor重启任务已提交"),
    ))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/supervisor/activate-timeout",
    tag = "Supervisor模块",
    request_body = TimeoutRequest,
    description = "传递分钟激活超时沙箱销毁设置，并关闭自动保活配置",
    responses(
        (status = 200, description = "超时销毁已设置", body = ApiResponse<SupervisorTimeout>),
        (status = 400, description = "超时时间无效", body = ApiResponse),
        (status = 500, description = "设置超时销毁失败", body = ApiResponse),
    ),
)]
async fn activate_timeout(
    State(state): State<AppState>,
    Json(request): Json<TimeoutRequest>,
) -> Result<ApiResponse<SupervisorTimeout>, AppException> {
    let result = state.supervisor_service.activate_timeout(request.minutes)?;

    let timeout_minutes = result.timeout_minutes.unwrap_or_default();
    let message = format!("超时销毁已设置, 所有服务与沙箱将在{timeout_minutes}分钟后销毁");
    Ok(ApiResponse::success(Some(result), message))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/supervisor/extend-timeout",
    tag = "Supervisor模块",
    request_body = TimeoutRequest,
    description = "传递指定的分钟延长超时时间并关闭自动保活",
    responses(
        (status = 200, description = "超时销毁时间已延长", body = ApiResponse<SupervisorTimeout>),
        (status = 400, description = "超时销毁未激活或分钟数无效", body = ApiResponse),
        (status = 500, description = "延长超时销毁失败", body = ApiResponse),
    ),
)]
async fn extend_timeout(
    State(state): State<AppState>,
    Json(request): Json<TimeoutRequest>,
) -> Result<ApiResponse<SupervisorTimeout>, AppException> {
    let extension_minutes = request.minutes.unwrap_or(DEFAULT_TIMEOUT_EXTENSION_MINUTES);
    let result = state.supervisor_service.extend_timeout(request.minutes)?;
    let timeout_minutes = result.timeout_minutes.unwrap_or_default();
    let message = format!(
        "超时销毁时间已延长{extension_minutes}分钟, 所有服务与沙箱将在{timeout_minutes}分钟后销毁"
    );
    Ok(ApiResponse::success(Some(result), message))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/supervisor/cancel-timeout",
    tag = "Supervisor模块",
    description = "取消超时销毁配置",
    responses(
        (status = 200, description = "超时销毁已取消或尚未激活", body = ApiResponse<SupervisorTimeout>),
    ),
)]
async fn cancel_timeout(State(state): State<AppState>) -> ApiResponse<SupervisorTimeout> {
    let result = state.supervisor_service.cancel_timeout();
    let message = if result.status.as_deref() == Some("timeout_cancelled") {
        "超时销毁已取消"
    } else {
        "超时销毁未激活"
    };
    ApiResponse::success(Some(result), message)
}

#[utoipa::path(
    get,
    context_path = "/api",
    path = "/supervisor/timeout-status",
    tag = "Supervisor模块",
    description = "获取当前supervisor的超时状态配置",
    responses(
        (status = 200, description = "获取超时销毁状态成功", body = ApiResponse<SupervisorTimeout>),
    ),
)]
async fn get_timeout_status(State(state): State<AppState>) -> ApiResponse<SupervisorTimeout> {
    let result = state.supervisor_service.get_timeout_status();
    let message = if result.active {
        format!(
            "剩余超时销毁分钟数: {}",
            result.remaining_seconds.unwrap_or_default() / 60
        )
    } else {
        "未激活超时销毁".to_string()
    };
    ApiResponse::success(Some(result), message)
}

impl SupervisorController {
    pub fn routes() -> Router<AppState> {
        Router::new()
            .route("/status", get(get_status))
            .route("/stop-all-processes", post(stop_all_processes))
            .route("/shutdown", post(shutdown))
            .route("/restart", post(restart))
            .route("/activate-timeout", post(activate_timeout))
            .route("/extend-timeout", post(extend_timeout))
            .route("/cancel-timeout", post(cancel_timeout))
            .route("/timeout-status", get(get_timeout_status))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use utoipa::OpenApi;

    use super::*;

    #[test]
    fn openapi_exposes_supervisor_paths_and_restart_responses() {
        let openapi = crate::controllers::ApiDoc::openapi();

        assert!(openapi.paths.paths.contains_key("/api/supervisor/status"));
        assert!(
            openapi
                .paths
                .paths
                .contains_key("/api/supervisor/stop-all-processes")
        );
        assert!(openapi.paths.paths.contains_key("/api/supervisor/shutdown"));
        assert!(openapi.paths.paths.contains_key("/api/supervisor/restart"));
        assert!(
            openapi
                .paths
                .paths
                .contains_key("/api/supervisor/activate-timeout")
        );
        assert!(
            openapi
                .paths
                .paths
                .contains_key("/api/supervisor/extend-timeout")
        );
        assert!(
            openapi
                .paths
                .paths
                .contains_key("/api/supervisor/cancel-timeout")
        );
        assert!(
            openapi
                .paths
                .paths
                .contains_key("/api/supervisor/timeout-status")
        );

        let restart = openapi
            .paths
            .paths
            .get("/api/supervisor/restart")
            .and_then(|item| item.post.as_ref())
            .expect("restart POST operation should exist");
        assert!(restart.responses.responses.contains_key("202"));
        assert!(restart.responses.responses.contains_key("400"));
    }

    #[tokio::test]
    async fn timeout_controllers_manage_the_manual_and_automatic_modes() {
        let supervisor_service = Arc::new(crate::services::SupervisorService::new());
        let state = AppState {
            shell_service: Arc::new(crate::services::ShellService::new()),
            file_service: Arc::new(crate::services::FileService::new()),
            supervisor_service: Arc::clone(&supervisor_service),
        };

        let activated = activate_timeout(
            State(state.clone()),
            Json(TimeoutRequest { minutes: Some(10) }),
        )
        .await
        .expect("activate endpoint should succeed");
        assert_eq!(activated.data.status.as_deref(), Some("timeout_activated"));
        assert_eq!(activated.data.timeout_minutes, Some(10));
        assert!(!supervisor_service.expand_enabled());

        let extended = extend_timeout(
            State(state.clone()),
            Json(TimeoutRequest { minutes: Some(5) }),
        )
        .await
        .expect("extend endpoint should succeed");
        assert_eq!(extended.data.status.as_deref(), Some("timeout_extended"));
        assert_eq!(
            extended.msg,
            "超时销毁时间已延长5分钟, 所有服务与沙箱将在15分钟后销毁"
        );
        assert!(!supervisor_service.expand_enabled());

        let cancelled = cancel_timeout(State(state.clone())).await;
        assert_eq!(cancelled.data.status.as_deref(), Some("timeout_cancelled"));
        assert!(supervisor_service.expand_enabled());

        let status = get_timeout_status(State(state)).await;
        assert!(!status.data.active);
        assert_eq!(status.msg, "未激活超时销毁");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_status_wraps_empty_process_list_with_course_message() {
        use tokio::{
            io::{AsyncReadExt, AsyncWriteExt},
            net::UnixListener,
        };

        let socket_path = format!(
            "/tmp/sandbox-supervisor-controller-test-{}.sock",
            uuid::Uuid::new_v4()
        );
        let listener = UnixListener::bind(&socket_path).expect("test socket should bind");
        let body = r#"<?xml version="1.0"?><methodResponse><params><param><value><array><data></data></array></value></param></params></methodResponse>"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("client should connect");
            let mut request = vec![0_u8; 4096];
            let _ = stream
                .read(&mut request)
                .await
                .expect("request should read");
            stream
                .write_all(response.as_bytes())
                .await
                .expect("response should write");
        });

        let state = AppState {
            shell_service: Arc::new(crate::services::ShellService::new()),
            file_service: Arc::new(crate::services::FileService::new()),
            supervisor_service: Arc::new(crate::services::SupervisorService::with_rpc_url(
                socket_path.clone(),
            )),
        };

        let response = get_status(State(state))
            .await
            .expect("empty process list should be successful");

        assert_eq!(response.code, 200);
        assert_eq!(response.msg, "获取沙箱进程服务成功");
        assert!(response.data.is_empty());
        server.await.expect("mock server should finish");
        std::fs::remove_file(socket_path).expect("test socket should be removed");
    }
}
