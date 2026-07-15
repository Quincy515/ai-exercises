use std::path::{Path, PathBuf};

use axum::{
    Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, Multipart, Query, State},
    http::{Response, header},
    routing::{get, post},
};
use tokio::{fs, io::AsyncWriteExt};
use tokio_util::io::ReaderStream;
use uuid::Uuid;

use crate::{
    exceptions::{ApiResponse, AppException},
    models::{
        FileCheckResult, FileDeleteResult, FileFindResult, FileReadResult, FileReplaceResult,
        FileSearchResult, FileUploadResult, FileWriteResult,
    },
    views::{
        FileCheckRequest, FileDeleteRequest, FileDownloadRequest, FileFindRequest, FileReadRequest,
        FileReplaceRequest, FileSearchRequest, FileUploadForm, FileWriteRequest,
    },
};

use super::service_dependencies::AppState;

pub struct FileController;

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/file/read-file",
    tag = "文件模块",
    description = "根据传递的数据读取沙箱中的文件内容",
    request_body = FileReadRequest,
    responses(
        (status = 200, description = "成功", body = ApiResponse<FileReadResult>),
        (status = 400, description = "请求错误", body = ApiResponse),
        (status = 404, description = "文件不存在或无权限", body = ApiResponse),
    ),
)]
async fn read_file(
    State(state): State<AppState>,
    Json(request): Json<FileReadRequest>,
) -> Result<ApiResponse<FileReadResult>, AppException> {
    let result = state
        .file_service
        .read_file(
            &request.file_path,
            request.start_line,
            request.end_line,
            request.sudo,
            request.max_length,
        )
        .await?;

    Ok(ApiResponse::success(Some(result), "文件内容读取成功"))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/file/write-file",
    tag = "文件模块",
    description = "根据传递的数据向指定文件写入内容",
    request_body = FileWriteRequest,
    responses(
        (status = 200, description = "成功", body = ApiResponse<FileWriteResult>),
        (status = 400, description = "请求错误", body = ApiResponse),
        (status = 500, description = "文件写入失败", body = ApiResponse),
    ),
)]
async fn write_file(
    State(state): State<AppState>,
    Json(request): Json<FileWriteRequest>,
) -> Result<ApiResponse<FileWriteResult>, AppException> {
    let result = state
        .file_service
        .write_file(
            &request.file_path,
            request.content,
            request.append,
            request.leading_newline,
            request.trailing_newline,
            request.sudo,
        )
        .await?;

    Ok(ApiResponse::success(Some(result), "文件内容写入成功"))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/file/replace-in-file",
    tag = "文件模块",
    description = "根据传递的数据替换文件内的部分内容",
    request_body = FileReplaceRequest,
    responses(
        (status = 200, description = "成功", body = ApiResponse<FileReplaceResult>),
        (status = 400, description = "请求错误", body = ApiResponse),
        (status = 404, description = "文件不存在或无权限", body = ApiResponse),
        (status = 500, description = "文件替换失败", body = ApiResponse),
    ),
)]
async fn replace_in_file(
    State(state): State<AppState>,
    Json(request): Json<FileReplaceRequest>,
) -> Result<ApiResponse<FileReplaceResult>, AppException> {
    let result = state
        .file_service
        .replace_in_file(
            &request.file_path,
            &request.old_str,
            &request.new_str,
            request.sudo,
        )
        .await?;
    let message = format!("文件内容替换完成, 已替换{}处内容", result.replaced_count);

    Ok(ApiResponse::success(Some(result), message))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/file/search-in-file",
    tag = "文件模块",
    description = "根据传递的数据检索指定文件的内容",
    request_body = FileSearchRequest,
    responses(
        (status = 200, description = "成功", body = ApiResponse<FileSearchResult>),
        (status = 400, description = "正则表达式或请求错误", body = ApiResponse),
        (status = 404, description = "文件不存在或无权限", body = ApiResponse),
    ),
)]
async fn search_in_file(
    State(state): State<AppState>,
    Json(request): Json<FileSearchRequest>,
) -> Result<ApiResponse<FileSearchResult>, AppException> {
    let result = state
        .file_service
        .search_in_file(&request.file_path, &request.regex, request.sudo)
        .await?;
    let message = format!("文件内容搜索完成, 找到{}处匹配内容", result.matches.len());

    Ok(ApiResponse::success(Some(result), message))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/file/find-files",
    tag = "文件模块",
    description = "根据传递的文件夹和glob文件规则查找文件列表",
    request_body = FileFindRequest,
    responses(
        (status = 200, description = "成功", body = ApiResponse<FileFindResult>),
        (status = 400, description = "glob规则错误", body = ApiResponse),
        (status = 404, description = "目录不存在", body = ApiResponse),
    ),
)]
async fn find_files(
    State(state): State<AppState>,
    Json(request): Json<FileFindRequest>,
) -> Result<ApiResponse<FileFindResult>, AppException> {
    let result = state
        .file_service
        .find_files(&request.dir_path, &request.glob_pattern)
        .await?;
    let message = format!("查找完毕, 检索到{}个文件", result.files.len());

    Ok(ApiResponse::success(Some(result), message))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/file/upload-file",
    tag = "文件模块",
    description = "根据传递的文件源和路径上传文件到沙箱",
    request_body(content = inline(FileUploadForm), content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "成功", body = ApiResponse<FileUploadResult>),
        (status = 400, description = "上传表单错误", body = ApiResponse),
        (status = 500, description = "文件上传失败", body = ApiResponse),
    ),
)]
async fn upload_file(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<ApiResponse<FileUploadResult>, AppException> {
    let (upload, requested_path) = receive_upload(multipart).await?;
    let file_path = requested_path.unwrap_or_else(|| {
        std::env::temp_dir()
            .join(&upload.file_name)
            .to_string_lossy()
            .into_owned()
    });
    let result = state
        .file_service
        .upload_file(&upload.temp_path, &file_path)
        .await?;

    Ok(ApiResponse::success(Some(result), "文件上传成功"))
}

#[utoipa::path(
    get,
    context_path = "/api",
    path = "/file/download-file",
    tag = "文件模块",
    description = "根据传递的file_path下载指定文件",
    params(FileDownloadRequest),
    responses(
        (status = 200, description = "文件流", body = Vec<u8>, content_type = "application/octet-stream",
            headers(("content-disposition" = String, description = "附件文件名"))),
        (status = 404, description = "文件不存在", body = ApiResponse),
        (status = 500, description = "文件下载失败", body = ApiResponse),
    ),
)]
async fn download_file(
    State(state): State<AppState>,
    Query(request): Query<FileDownloadRequest>,
) -> Result<Response<Body>, AppException> {
    // 1.确保下当前文件存在
    state.file_service.ensure_file(&request.file_path).await?;

    // 2.打开文件并读取内容
    let file = fs::File::open(&request.file_path)
        .await
        .map_err(|err| AppException::internal(format!("下载文件失败: {err}")))?;
    let file_size = file
        .metadata()
        .await
        .map_err(|err| AppException::internal(format!("读取下载文件信息失败: {err}")))?
        .len();
    let file_name = Path::new(&request.file_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");

    // 3.返回文件下载响应
    Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, file_size)
        .header(
            header::CONTENT_DISPOSITION,
            attachment_content_disposition(file_name),
        )
        .body(Body::from_stream(ReaderStream::new(file)))
        .map_err(|err| AppException::internal(format!("创建文件下载响应失败: {err}")))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/file/check-file-exists",
    tag = "文件模块",
    description = "根据传递的路径判断文件是否存在",
    request_body = FileCheckRequest,
    responses(
        (status = 200, description = "成功", body = ApiResponse<FileCheckResult>),
        (status = 500, description = "文件检查失败", body = ApiResponse),
    ),
)]
async fn check_file_exists(
    State(state): State<AppState>,
    Json(request): Json<FileCheckRequest>,
) -> Result<ApiResponse<FileCheckResult>, AppException> {
    let result = state
        .file_service
        .check_file_exists(&request.file_path)
        .await?;
    let message = if result.exists {
        "文件存在"
    } else {
        "文件不存在"
    };

    Ok(ApiResponse::success(Some(result), message))
}

#[utoipa::path(
    post,
    context_path = "/api",
    path = "/file/delete-file",
    tag = "文件模块",
    description = "根据传递的文件路径删除指定文件",
    request_body = FileDeleteRequest,
    responses(
        (status = 200, description = "成功", body = ApiResponse<FileDeleteResult>),
        (status = 404, description = "文件不存在", body = ApiResponse),
        (status = 500, description = "文件删除失败", body = ApiResponse),
    ),
)]
async fn delete_file(
    State(state): State<AppState>,
    Json(request): Json<FileDeleteRequest>,
) -> Result<ApiResponse<FileDeleteResult>, AppException> {
    let result = state.file_service.delete_file(&request.file_path).await?;

    Ok(ApiResponse::success(Some(result), "删除文件成功"))
}

struct PendingUpload {
    temp_path: PathBuf,
    file_name: String,
}

impl Drop for PendingUpload {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.temp_path);
    }
}

async fn receive_upload(
    mut multipart: Multipart,
) -> Result<(PendingUpload, Option<String>), AppException> {
    let mut upload = None;
    let mut requested_path = None;

    while let Some(mut field) = multipart
        .next_field()
        .await
        .map_err(|err| AppException::bad_request(format!("读取上传表单失败: {err}")))?
    {
        let field_name = field.name().map(str::to_owned);
        match field_name.as_deref() {
            Some("file") => {
                if upload.is_some() {
                    return Err(AppException::bad_request("上传表单只能包含一个file字段"));
                }

                let file_name = safe_upload_file_name(field.file_name().unwrap_or("upload.bin"));
                let pending = PendingUpload {
                    temp_path: std::env::temp_dir()
                        .join(format!("sandbox-upload-{}", Uuid::new_v4())),
                    file_name,
                };
                let mut output = fs::File::create(&pending.temp_path).await.map_err(|err| {
                    AppException::internal(format!("创建上传临时文件失败: {err}"))
                })?;

                while let Some(chunk) = field
                    .chunk()
                    .await
                    .map_err(|err| AppException::bad_request(format!("读取上传文件失败: {err}")))?
                {
                    output.write_all(&chunk).await.map_err(|err| {
                        AppException::internal(format!("写入上传临时文件失败: {err}"))
                    })?;
                }
                output.flush().await.map_err(|err| {
                    AppException::internal(format!("刷新上传临时文件失败: {err}"))
                })?;
                upload = Some(pending);
            }
            Some("file_path" | "filepath") => {
                let value = field
                    .text()
                    .await
                    .map_err(|err| AppException::bad_request(format!("读取上传路径失败: {err}")))?;
                let value = value.trim();
                if !value.is_empty() {
                    requested_path = Some(value.to_string());
                }
            }
            _ => {}
        }
    }

    let upload = upload.ok_or_else(|| AppException::bad_request("上传表单缺少file字段"))?;
    Ok((upload, requested_path))
}

fn safe_upload_file_name(file_name: &str) -> String {
    file_name
        .rsplit(['/', '\\'])
        .find(|part| !part.is_empty() && !matches!(*part, "." | ".."))
        .unwrap_or("upload.bin")
        .to_string()
}

fn attachment_content_disposition(file_name: &str) -> String {
    let fallback = file_name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    let encoded = rfc5987_encode(file_name);

    format!("attachment; filename=\"{fallback}\"; filename*=UTF-8''{encoded}")
}

fn rfc5987_encode(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(value.len());

    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[(byte >> 4) as usize]));
            encoded.push(char::from(HEX[(byte & 0x0f) as usize]));
        }
    }

    encoded
}

impl FileController {
    pub fn routes() -> Router<AppState> {
        Router::new()
            .route("/read-file", post(read_file))
            .route("/write-file", post(write_file))
            .route("/replace-in-file", post(replace_in_file))
            .route("/search-in-file", post(search_in_file))
            .route("/find-files", post(find_files))
            .route(
                "/upload-file",
                post(upload_file).layer(DefaultBodyLimit::disable()),
            )
            .route("/download-file", get(download_file))
            .route("/check-file-exists", post(check_file_exists))
            .route("/delete-file", post(delete_file))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use axum::{
        body::{Body, to_bytes},
        extract::{FromRequest, Multipart, Query},
        http::{Request, header},
    };
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
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

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!("sandbox-controller-dir-{}", Uuid::new_v4())))
        }

        fn file_path(&self, relative_path: &str) -> PathBuf {
            self.0.join(relative_path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[tokio::test]
    async fn returns_the_file_content_in_a_unified_response() {
        let file = TestFile::create("Agent读取文件").await;

        let response = read_file(
            State(AppState::new()),
            Json(FileReadRequest {
                file_path: file.as_str().to_string(),
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
        assert_eq!(response.data.file_path, file.as_str());
        assert_eq!(response.data.content, "Agent读取文件");
    }

    #[tokio::test]
    async fn writes_file_content_and_returns_a_unified_response() {
        let file = TestFile::create("旧内容").await;

        let response = write_file(
            State(AppState::new()),
            Json(FileWriteRequest {
                file_path: file.as_str().to_string(),
                content: "Agent写入文件".to_string(),
                append: Some(false),
                leading_newline: Some(false),
                trailing_newline: Some(true),
                sudo: Some(false),
            }),
        )
        .await
        .expect("controller should write file content");

        let expected = "Agent写入文件\n";
        assert_eq!(response.code, 200);
        assert_eq!(response.msg, "文件内容写入成功");
        assert_eq!(response.data.file_path, file.as_str());
        assert_eq!(response.data.bytes_written, Some(expected.len()));
        assert_eq!(tokio::fs::read_to_string(&file.0).await.unwrap(), expected);
    }

    #[tokio::test]
    async fn replaces_file_content_and_reports_the_replacement_count() {
        let file = TestFile::create("Agent开发 Agent工具").await;

        let response = replace_in_file(
            State(AppState::new()),
            Json(FileReplaceRequest {
                file_path: file.as_str().to_string(),
                old_str: "Agent".to_string(),
                new_str: "Manus".to_string(),
                sudo: Some(false),
            }),
        )
        .await
        .expect("controller should replace matching content");

        assert_eq!(response.msg, "文件内容替换完成, 已替换2处内容");
        assert_eq!(response.data.replaced_count, 2);
        assert_eq!(
            tokio::fs::read_to_string(&file.0).await.unwrap(),
            "Manus开发 Manus工具"
        );
    }

    #[tokio::test]
    async fn searches_content_and_finds_files_with_count_messages() {
        let directory = TestDirectory::new();
        let file_path = directory.file_path("src/agent.rs");
        tokio::fs::create_dir_all(file_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&file_path, "Agent开发\n工具开发\n")
            .await
            .unwrap();

        let searched = search_in_file(
            State(AppState::new()),
            Json(FileSearchRequest {
                file_path: file_path.to_string_lossy().into_owned(),
                regex: "Agent".to_string(),
                sudo: Some(false),
            }),
        )
        .await
        .expect("controller should search file content");
        let found = find_files(
            State(AppState::new()),
            Json(FileFindRequest {
                dir_path: directory.0.to_string_lossy().into_owned(),
                glob_pattern: "**/*.rs".to_string(),
            }),
        )
        .await
        .expect("controller should find files by glob rule");

        assert_eq!(searched.msg, "文件内容搜索完成, 找到1处匹配内容");
        assert_eq!(searched.data.line_numbers, [0]);
        assert_eq!(found.msg, "查找完毕, 检索到1个文件");
        assert_eq!(found.data.files, [file_path.to_string_lossy().into_owned()]);
    }

    #[tokio::test]
    async fn accepts_upload_path_after_the_file_field_and_keeps_streamed_bytes() {
        let directory = TestDirectory::new();
        let target = directory.file_path("nested/demo.txt");
        let target = target.to_str().unwrap();
        let boundary = "sandbox-upload-boundary";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"source.txt\"\r\nContent-Type: text/plain\r\n\r\nAgent upload\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file_path\"\r\n\r\n{target}\r\n--{boundary}--\r\n"
        );
        let request = Request::builder()
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap();
        let multipart = Multipart::from_request(request, &())
            .await
            .expect("multipart body should be extractable");

        let response = upload_file(State(AppState::new()), multipart)
            .await
            .expect("controller should upload multipart file");

        assert_eq!(response.data.file_path, target);
        assert_eq!(response.data.file_size, 12);
        assert_eq!(tokio::fs::read(target).await.unwrap(), b"Agent upload");
    }

    #[tokio::test]
    async fn upload_route_streams_files_larger_than_axums_default_body_limit() {
        let directory = TestDirectory::new();
        let target = directory.file_path("large.bin");
        let boundary = "sandbox-large-upload-boundary";
        let payload = vec![b'x'; 2 * 1024 * 1024 + 1];
        let mut body = Vec::with_capacity(payload.len() + 512);
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file_path\"\r\n\r\n{}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"large.bin\"\r\nContent-Type: application/octet-stream\r\n\r\n",
                target.to_string_lossy()
            )
            .as_bytes(),
        );
        body.extend_from_slice(&payload);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = FileController::routes().with_state(AppState::new());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        let request_head = format!(
            "POST /upload-file HTTP/1.1\r\nHost: {address}\r\nContent-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(request_head.as_bytes()).await.unwrap();
        stream.write_all(&body).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        server.abort();

        assert!(
            response.starts_with(b"HTTP/1.1 200 OK"),
            "unexpected response: {}",
            String::from_utf8_lossy(&response[..response.len().min(1024)])
        );
        assert_eq!(
            tokio::fs::metadata(&target).await.unwrap().len(),
            payload.len() as u64
        );
    }

    #[tokio::test]
    async fn downloads_the_file_as_an_attachment_stream() {
        let file = TestFile::create("下载内容").await;

        let response = download_file(
            State(AppState::new()),
            Query(FileDownloadRequest {
                file_path: file.as_str().to_string(),
            }),
        )
        .await
        .expect("controller should stream existing file");

        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/octet-stream"
        );
        assert!(
            response.headers()[header::CONTENT_DISPOSITION]
                .to_str()
                .unwrap()
                .starts_with("attachment;")
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body, "下载内容".as_bytes());
    }

    #[tokio::test]
    async fn checks_then_deletes_a_file_through_unified_responses() {
        let file = TestFile::create("待删除").await;

        let checked = check_file_exists(
            State(AppState::new()),
            Json(FileCheckRequest {
                file_path: file.as_str().to_string(),
            }),
        )
        .await
        .expect("controller should check file existence");
        let deleted = delete_file(
            State(AppState::new()),
            Json(FileDeleteRequest {
                file_path: file.as_str().to_string(),
            }),
        )
        .await
        .expect("controller should delete existing file");

        assert_eq!(checked.msg, "文件存在");
        assert!(checked.data.exists);
        assert_eq!(deleted.msg, "删除文件成功");
        assert!(deleted.data.deleted);
    }

    #[test]
    fn upload_file_name_keeps_only_a_safe_base_name() {
        assert_eq!(safe_upload_file_name(r"C:\\docs\\agent.txt"), "agent.txt");
        assert_eq!(safe_upload_file_name("../../agent.txt"), "agent.txt");
        assert_eq!(safe_upload_file_name(".."), "upload.bin");
        assert_eq!(safe_upload_file_name("."), "upload.bin");
    }
}
