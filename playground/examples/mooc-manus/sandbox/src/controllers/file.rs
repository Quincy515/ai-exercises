use axum::{Json, Router, extract::State, routing::post};

use crate::{
    exceptions::{ApiResponse, AppException},
    models::FileReadResult,
    views::ReadFileRequest,
};

use super::service_dependencies::AppState;

pub struct FileController;

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/file/read-file",
    tag = "文件模块",
    description = "根据传递的数据读取沙箱中的文件内容",
    request_body = ReadFileRequest,
    responses(
        (status = 200, description = "成功", body = ApiResponse<FileReadResult>),
        (status = 400, description = "请求错误", body = ApiResponse),
        (status = 404, description = "文件不存在或无权限", body = ApiResponse),
    ),
)]
async fn read_file(
    State(state): State<AppState>,
    Json(request): Json<ReadFileRequest>,
) -> Result<ApiResponse<FileReadResult>, AppException> {
    let result = state
        .file_service
        .read_file(
            &request.filepath,
            request.start_line,
            request.end_line,
            request.sudo,
            request.max_length,
        )
        .await?;

    Ok(ApiResponse::success(Some(result), "文件内容读取成功"))
}

impl FileController {
    pub fn routes() -> Router<AppState> {
        Router::new().route("/read-file", post(read_file))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use uuid::Uuid;

    use super::*;

    struct TestFile(PathBuf);

    impl TestFile {
        async fn create(content: &str) -> Self {
            let path = std::env::temp_dir().join(format!("sandbox-controller-{}", Uuid::new_v4()));
            tokio::fs::write(&path, content)
                .await
                .expect("test file should be writable");
            Self(path)
        }

        fn as_str(&self) -> &str {
            self.0.to_str().expect("test path should be utf-8")
        }
    }

    impl Drop for TestFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[tokio::test]
    async fn returns_the_file_content_in_a_unified_response() {
        let file = TestFile::create("Agent读取文件").await;

        let response = read_file(
            State(AppState::new()),
            Json(ReadFileRequest {
                filepath: file.as_str().to_string(),
                start_line: None,
                end_line: None,
                sudo: Some(false),
                max_length: Some(10_000),
            }),
        )
        .await
        .expect("controller should return file content");

        assert_eq!(response.code, 200);
        assert_eq!(response.msg, "文件内容读取成功");
        assert_eq!(response.data.filepath, file.as_str());
        assert_eq!(response.data.content, "Agent读取文件");
    }
}
