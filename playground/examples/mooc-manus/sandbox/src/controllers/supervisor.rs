use axum::{
    Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};

use crate::{
    exceptions::{ApiResponse, AppException},
    models::{ProcessInfo, SupervisorActionResult},
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

impl SupervisorController {
    pub fn routes() -> Router<AppState> {
        Router::new()
            .route("/status", get(get_status))
            .route("/stop-all-processes", post(stop_all_processes))
            .route("/shutdown", post(shutdown))
            .route("/restart", post(restart))
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

        let restart = openapi
            .paths
            .paths
            .get("/api/supervisor/restart")
            .and_then(|item| item.post.as_ref())
            .expect("restart POST operation should exist");
        assert!(restart.responses.responses.contains_key("202"));
        assert!(restart.responses.responses.contains_key("400"));
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
