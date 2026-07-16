use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};

use crate::{
    controllers::service_dependencies::AppState, services::DEFAULT_TIMEOUT_EXTENSION_MINUTES,
};

#[cfg(test)]
use crate::services::SupervisorService;

/// 这些接口会直接管理超时状态，访问它们时不能再由中间件自动增加时间。
const AUTO_EXTEND_IGNORED_PATHS: [&str; 4] = [
    "/api/supervisor/activate-timeout",
    "/api/supervisor/extend-timeout",
    "/api/supervisor/cancel-timeout",
    "/api/supervisor/timeout-status",
];

/// 使用中间件延长每次业务 API 请求的超时销毁时间。
pub async fn auto_extend_timeout_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path().to_string();

    // 1. 仅处理普通 `/api/` 业务请求，并排除四个直接管理超时状态的接口。
    if should_auto_extend_path(&path) {
        // 2. 自动续期失败只记录警告，原始业务请求仍应继续执行。
        match state
            .supervisor_service
            .auto_extend_timeout(DEFAULT_TIMEOUT_EXTENSION_MINUTES)
        {
            Ok(true) => {
                tracing::debug!(path, "调用 API 请求并自动延长超时销毁时长");
            }
            Ok(false) => {}
            Err(err) => {
                tracing::warn!(path, error = %err, "自动延长超时销毁时长失败");
            }
        }
    }

    // 3. 把请求交给后续中间件和路由处理器。
    next.run(request).await
}

fn should_auto_extend_path(path: &str) -> bool {
    path.starts_with("/api/")
        && !AUTO_EXTEND_IGNORED_PATHS
            .iter()
            .any(|ignored| path.starts_with(ignored))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{Router, http::StatusCode, routing::get};

    use super::*;

    fn app_state(supervisor_service: Arc<SupervisorService>) -> AppState {
        AppState {
            shell_service: Arc::new(crate::services::ShellService::new()),
            file_service: Arc::new(crate::services::FileService::new()),
            supervisor_service,
        }
    }

    #[test]
    fn middleware_only_considers_eligible_api_paths() {
        assert!(should_auto_extend_path("/api/shell/exec-command"));
        for ignored in AUTO_EXTEND_IGNORED_PATHS {
            assert!(!should_auto_extend_path(ignored));
        }
        assert!(!should_auto_extend_path("/api-docs/openapi.json"));
    }

    #[tokio::test]
    async fn axum_middleware_extends_an_eligible_request_by_three_minutes() {
        let supervisor_service = Arc::new(SupervisorService::new());
        let before = supervisor_service
            .shutdown_time()
            .expect("default timeout should be active");
        let app = Router::new()
            .route("/api/probe", get(|| async { StatusCode::NO_CONTENT }))
            .layer(axum::middleware::from_fn_with_state(
                app_state(Arc::clone(&supervisor_service)),
                auto_extend_timeout_middleware,
            ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener should bind");
        let address = listener
            .local_addr()
            .expect("test listener should have an address");
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("test app should serve");
        });

        let response = reqwest::get(format!("http://{address}/api/probe"))
            .await
            .expect("middleware test request should succeed");
        assert_eq!(response.status(), StatusCode::NO_CONTENT);

        let after = supervisor_service
            .shutdown_time()
            .expect("timeout should remain active");
        assert!(after >= before + std::time::Duration::from_secs(179));
        assert!(after <= before + std::time::Duration::from_secs(181));

        server.abort();
        let _ = server.await;
    }
}
