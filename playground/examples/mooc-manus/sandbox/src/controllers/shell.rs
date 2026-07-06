use std::env;

use axum::{Json, Router, extract::State, routing::post};

use crate::{
    controllers::service_dependencies::AppState,
    exceptions::{ApiResponse, AppException},
    models::{
        ShellExecuteResult, ShellKillResult, ShellReadResult, ShellWaitResult, ShellWriteResult,
    },
    views::{
        ShellExecuteRequest, ShellKillRequest, ShellReadRequest, ShellWaitRequest,
        ShellWriteRequest,
    },
};

const EMPTY_SESSION_MSG: &str = "Shell会话ID为空, 请核实后重试";

pub struct ShellController;

pub fn default_exec_dir() -> String {
    env::var("HOME")
        .or_else(|_| env::var("USERPROFILE"))
        .unwrap_or_else(|_| {
            env::current_dir()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|_| ".".to_string())
        })
}

fn require_session_id(session_id: String) -> Result<String, AppException> {
    if session_id.is_empty() {
        Err(AppException::bad_request(EMPTY_SESSION_MSG))
    } else {
        Ok(session_id)
    }
}

#[utoipa::path(
    post,
    path = "/shell/exec-command",
    tag = "Shell模块",
    request_body = ShellExecuteRequest,
    description = "在指定的 Shell 会话中运行命令",
    responses(
        (status = 200, description = "成功", body = ApiResponse<ShellExecuteResult>),
        (status = 400, description = "请求错误", body = ApiResponse),
    ),
)]
async fn exec_command(
    State(state): State<AppState>,
    Json(request): Json<ShellExecuteRequest>,
) -> Result<ApiResponse<ShellExecuteResult>, AppException> {
    // 1.判断下是否传递了 session_id，如果不存在则新建一个 session_id
    let session_id = match request.session_id {
        Some(session_id) if !session_id.is_empty() => session_id,
        _ => state.shell_service.create_session_id()?,
    };

    // 2.判断下是否传递了执行目录，如果未传递则使用根目录作为执行路径
    let exec_dir = match request.exec_dir {
        Some(exec_dir) if !exec_dir.is_empty() => exec_dir,
        _ => default_exec_dir(),
    };

    // 3.调用服务执行命令获取结果
    let result = state
        .shell_service
        .exec_command(session_id, exec_dir, request.command)
        .await?;

    Ok(ApiResponse::success_default(Some(result)))
}

#[utoipa::path(
    post,
    path = "/shell/read-shell-output",
    tag = "Shell模块",
    request_body = ShellReadRequest,
    description = "根据传递的会话 id+是否返回控制台标识获取 Shell 命令执行结果",
    responses(
        (status = 200, description = "成功", body = ApiResponse<ShellReadResult>),
        (status = 400, description = "请求错误", body = ApiResponse),
    ),
)]
async fn read_shell_output(
    State(state): State<AppState>,
    Json(request): Json<ShellReadRequest>,
) -> Result<ApiResponse<ShellReadResult>, AppException> {
    // 1.判断下 Shell 会话 id 是否存在
    let session_id = require_session_id(request.session_id)?;

    // 2.调用服务获取命令执行结果
    let result = state
        .shell_service
        .read_shell_output(session_id, request.console.unwrap_or(false))
        .await?;

    Ok(ApiResponse::success_default(Some(result)))
}

#[utoipa::path(
    post,
    path = "/shell/wait-process",
    tag = "Shell模块",
    request_body = ShellWaitRequest,
    description = "传递会话 id+描述执行等待并获取等待结果",
    responses(
        (status = 200, description = "成功", body = ApiResponse<ShellWaitResult>),
        (status = 400, description = "请求错误", body = ApiResponse),
    ),
)]
async fn wait_process(
    State(state): State<AppState>,
    Json(request): Json<ShellWaitRequest>,
) -> Result<ApiResponse<ShellWaitResult>, AppException> {
    // 1.判断下 Shell 会话 id 是否存在
    let session_id = require_session_id(request.session_id)?;

    // 2.调用服务等待子进程
    let result = state
        .shell_service
        .wait_process(session_id, request.seconds)
        .await?;

    Ok(ApiResponse::success(
        Some(result.clone()),
        format!("进程结束, 返回状态码(returncode): {}", result.returncode),
    ))
}

#[utoipa::path(
    post,
    path = "/shell/write-shell-input",
    tag = "Shell模块",
    request_body = ShellWriteRequest,
    description = "根据传递的会话+写入内容+按下回车标识向指定子进程写入数据",
    responses(
        (status = 200, description = "成功", body = ApiResponse<ShellWriteResult>),
        (status = 400, description = "请求错误", body = ApiResponse),
    ),
)]
async fn write_shell_input(
    State(state): State<AppState>,
    Json(request): Json<ShellWriteRequest>,
) -> Result<ApiResponse<ShellWriteResult>, AppException> {
    // 1.判断下 Shell 会话 id 是否存在
    let session_id = require_session_id(request.session_id)?;

    // 2.调用服务向子进程写入数据
    let result = state
        .shell_service
        .write_shell_input(session_id, request.input_text, request.press_enter)
        .await?;

    Ok(ApiResponse::success(Some(result), "向进程写入数据成功"))
}

#[utoipa::path(
    post,
    path = "/shell/kill-process",
    tag = "Shell模块",
    request_body = ShellKillRequest,
    description = "传递 Shell 会话 id 关闭指定会话",
    responses(
        (status = 200, description = "成功", body = ApiResponse<ShellKillResult>),
        (status = 400, description = "请求错误", body = ApiResponse),
    ),
)]
async fn kill_process(
    State(state): State<AppState>,
    Json(request): Json<ShellKillRequest>,
) -> Result<ApiResponse<ShellKillResult>, AppException> {
    // 1.判断下 Shell 会话 id 是否存在
    let session_id = require_session_id(request.session_id)?;

    // 2.调用服务关闭 Shell 会话
    let result = state.shell_service.kill_process(session_id).await?;
    let msg = if result.status == "terminated" {
        "进程终止"
    } else {
        "进程已结束"
    };

    Ok(ApiResponse::success(Some(result), msg))
}

impl ShellController {
    pub fn routes() -> Router<AppState> {
        Router::new()
            .route("/exec-command", post(exec_command))
            .route("/read-shell-output", post(read_shell_output))
            .route("/wait-process", post(wait_process))
            .route("/write-shell-input", post(write_shell_input))
            .route("/kill-process", post(kill_process))
    }
}

#[cfg(test)]
mod tests {
    use axum::{Json, extract::State};

    use super::*;

    #[tokio::test]
    async fn exec_command_fills_python_default_fields() {
        let response = exec_command(
            State(AppState::new()),
            Json(ShellExecuteRequest {
                session_id: None,
                exec_dir: Some(String::new()),
                command: "pwd".to_string(),
            }),
        )
        .await
        .expect("exec command should succeed");

        assert_eq!(response.data.command, "pwd");
        assert_eq!(response.data.status, "completed");
        assert_eq!(response.data.returncode, Some(0));
        let expected_exec_dir = default_exec_dir();
        assert_eq!(
            response.data.output.as_deref().map(str::trim),
            Some(expected_exec_dir.as_str())
        );
        assert!(!response.data.session_id.is_empty());
    }

    #[tokio::test]
    async fn exec_command_preserves_supplied_fields() {
        let response = exec_command(
            State(AppState::new()),
            Json(ShellExecuteRequest {
                session_id: Some("session-1".to_string()),
                exec_dir: Some("/tmp".to_string()),
                command: "printf preserved".to_string(),
            }),
        )
        .await
        .expect("exec command should succeed");

        assert_eq!(response.data.session_id, "session-1");
        assert_eq!(response.data.command, "printf preserved");
        assert_eq!(response.data.status, "completed");
        assert_eq!(response.data.output.as_deref(), Some("preserved"));
    }

    #[tokio::test]
    async fn read_shell_output_can_return_console_records() {
        let state = AppState::new();
        let session_id = state.shell_service.create_session_id().unwrap();

        exec_command(
            State(state.clone()),
            Json(ShellExecuteRequest {
                session_id: Some(session_id.clone()),
                exec_dir: Some("/tmp".to_string()),
                command: "printf console".to_string(),
            }),
        )
        .await
        .expect("exec command should succeed");

        let response = read_shell_output(
            State(state),
            Json(ShellReadRequest {
                session_id,
                console: Some(true),
            }),
        )
        .await
        .expect("read shell output should succeed");

        assert_eq!(response.data.output, "console");
        assert_eq!(response.data.console_records.len(), 1);
    }
}
