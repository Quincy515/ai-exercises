use axum::{
    body::Body,
    extract::{
        multipart::{MultipartError, MultipartRejection},
        DefaultBodyLimit, Multipart,
    },
    http::{header, StatusCode},
};
use futures::{StreamExt, TryStreamExt};
use loco_openapi::prelude::{openapi, routes};
use loco_rs::prelude::*;
use percent_encoding::{utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};

use crate::{
    application::error::AppError,
    domain::external::{FileNotFound, InvalidFileUpload, UploadFile},
    interfaces::service_dependencies::get_file_service,
    views::files::{FileResponse, FileUploadRequest},
};

/// 整个 multipart 请求最多 32 MiB，包含文件和表单边界；避免完整缓冲时无限占用内存。
pub const MAX_UPLOAD_BODY_SIZE: usize = 32 * 1024 * 1024;

// filename* 使用 UTF-8 百分号编码，只保留 URI 非保留字符。
const FILENAME_ENCODE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~');

/// 文件上传接口，传递文件返回文件的 File 信息。
#[utoipa::path(
    post,
    path = "/api/files",
    tag = "文件模块",
    summary = "对话文件上传接口",
    description = "将单个文件上传到对象存储并记录文件信息；表单字段为 file，请求上限 32 MiB。",
    request_body(content = FileUploadRequest, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "上传文件成功", body = FileResponse),
        (status = 400, description = "文件或 multipart 请求无效"),
        (status = 413, description = "上传请求超过大小限制"),
        (status = 500, description = "文件上传或信息保存失败")
    )
)]
#[debug_handler]
pub async fn upload_file(
    State(ctx): State<AppContext>,
    multipart: std::result::Result<Multipart, MultipartRejection>,
) -> Result<Response> {
    let mut multipart = multipart.map_err(|error| {
        AppError::business(
            error.status(),
            "file.invalid_multipart",
            error.body_text(),
            None,
        )
    })?;

    // 1.读取名为 file 的文件源；先保存借用的文件名、类型，再消费字段中的二进制内容。
    let mut upload = None;
    while let Some(field) = multipart.next_field().await.map_err(map_multipart_error)? {
        if field.name() != Some("file") {
            continue;
        }
        if upload.is_some() {
            return Err(
                AppError::bad_request("file.multiple_files", "每次请求只能上传一个文件").into(),
            );
        }
        let filename = field.file_name().map(str::to_owned).ok_or_else(|| {
            AppError::bad_request("file.missing_filename", "file 字段必须包含文件名")
        })?;
        let mime_type = field.content_type().map(str::to_owned);
        let content = field.bytes().await.map_err(map_multipart_error)?;
        upload = Some(UploadFile {
            filename,
            mime_type,
            content,
        });
    }
    let upload = upload
        .ok_or_else(|| AppError::bad_request("file.missing_file", "请通过 file 字段上传文件"))?;

    // 2.完整解析请求后再调用服务，避免表单后半段出错时已经写入文件。
    let file_service = get_file_service(&ctx);
    let fileinfo = file_service
        .upload_file(upload)
        .await
        .map_err(|error| map_file_error(error, "file.upload_failed"))?;
    format::json(FileResponse::success("上传文件成功", fileinfo))
}

/// 获取指定文件的基础信息。
#[utoipa::path(
    get,
    path = "/api/files/{file_id}",
    tag = "文件模块",
    summary = "获取文件信息接口",
    description = "根据文件业务 UUID 获取文件的基础信息。",
    params(("file_id" = String, Path, description = "文件业务 UUID")),
    responses(
        (status = 200, description = "获取文件信息成功", body = FileResponse),
        (status = 400, description = "文件 UUID 无效"),
        (status = 404, description = "文件不存在"),
        (status = 500, description = "文件信息读取失败")
    )
)]
#[debug_handler]
pub async fn get_file_info(
    State(ctx): State<AppContext>,
    Path(file_id): Path<String>,
) -> Result<Response> {
    validate_file_id(&file_id)?;
    let fileinfo = get_file_service(&ctx)
        .get_file_info(&file_id)
        .await
        .map_err(|error| map_file_error(error, "file.get_info_failed"))?;
    format::json(FileResponse::success("获取文件信息成功", fileinfo))
}

/// 下载指定的文件。
#[utoipa::path(
    get,
    path = "/api/files/{file_id}/download",
    tag = "文件模块",
    summary = "文件下载接口",
    description = "从对象存储中流式下载指定文件到本地，保留原始文件名。",
    params(("file_id" = String, Path, description = "文件业务 UUID")),
    responses(
        (status = 200, description = "文件二进制内容", body = String, content_type = "application/octet-stream",
            headers(
                ("Content-Disposition" = String, description = "包含 UTF-8 编码原始文件名的附件头"),
                ("Content-Length" = u64, description = "文件字节数")
            )),
        (status = 400, description = "文件 UUID 无效"),
        (status = 404, description = "文件不存在"),
        (status = 500, description = "文件下载失败")
    )
)]
#[debug_handler]
pub async fn download_file(
    State(ctx): State<AppContext>,
    Path(file_id): Path<String>,
) -> Result<Response> {
    validate_file_id(&file_id)?;
    // 1.调用服务获取文件源数据。
    let (mut file_data, fileinfo) = get_file_service(&ctx)
        .download_file(&file_id)
        .await
        .map_err(|error| map_file_error(error, "file.download_failed"))?;

    // 存储流可能延迟到首次读取才访问对象；先读一块，避免对象缺失时先发送 200。
    let first_chunk = file_data
        .try_next()
        .await
        .map_err(|error| map_file_error(error.into(), "file.download_failed"))?;
    let file_data =
        futures::stream::iter(first_chunk.map(Ok::<_, std::io::Error>)).chain(file_data);

    // 2.对文件中的中文名字进行 URL 编码。
    let encoded_filename = utf8_percent_encode(&fileinfo.filename, FILENAME_ENCODE_SET);
    let mime_type = if fileinfo.mime_type.is_empty() {
        "application/octet-stream"
    } else {
        &fileinfo.mime_type
    };

    // 3.返回文件流数据；不把整个下载文件收集进内存。
    Response::builder()
        .header(header::CONTENT_TYPE, mime_type)
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename*=utf-8''{encoded_filename}"),
        )
        .header(header::CONTENT_LENGTH, fileinfo.size.to_string())
        .body(Body::from_stream(file_data))
        .map_err(|error| {
            AppError::internal("file.invalid_download_headers", error.to_string()).into()
        })
}

fn validate_file_id(file_id: &str) -> std::result::Result<(), AppError> {
    uuid::Uuid::parse_str(file_id)
        .map(|_| ())
        .map_err(|_| AppError::bad_request("file.invalid_id", "文件 id 必须是有效的 UUID"))
}

fn map_multipart_error(error: MultipartError) -> AppError {
    AppError::business(
        error.status(),
        "file.invalid_multipart",
        error.body_text(),
        None,
    )
}

fn map_file_error(error: anyhow::Error, code: &'static str) -> AppError {
    if error.is::<FileNotFound>() {
        AppError::business(
            StatusCode::NOT_FOUND,
            "file.not_found",
            error.to_string(),
            None,
        )
    } else if error.is::<InvalidFileUpload>() {
        AppError::bad_request("file.invalid_upload", error.to_string())
    } else {
        AppError::internal(code, format!("{error:#}"))
    }
}

pub fn routes() -> Routes {
    Routes::new()
        .prefix("/api/files")
        .add(
            "/",
            openapi(post(upload_file), routes!(upload_file))
                .layer(DefaultBodyLimit::max(MAX_UPLOAD_BODY_SIZE)),
        )
        .add(
            "/{file_id}",
            openapi(get(get_file_info), routes!(get_file_info)),
        )
        .add(
            "/{file_id}/download",
            openapi(get(download_file), routes!(download_file)),
        )
}
