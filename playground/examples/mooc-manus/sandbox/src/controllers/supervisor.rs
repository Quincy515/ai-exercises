use axum::{Router, extract::State, routing::get};

use crate::exceptions::{ApiResponse, AppException};
use crate::models::ProcessInfo;

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

impl SupervisorController {
    pub fn routes() -> Router<AppState> {
        Router::new().route("/status", get(get_status))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use utoipa::OpenApi;

    use super::*;

    #[test]
    fn openapi_exposes_course_status_path() {
        let openapi = crate::controllers::ApiDoc::openapi();

        assert!(openapi.paths.paths.contains_key("/api/supervisor/status"));
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
            supervisor_service: Arc::new(crate::services::SupervisorService {
                rpc_url: socket_path.clone(),
            }),
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
