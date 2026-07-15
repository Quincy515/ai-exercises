use std::{any::Any, collections::BTreeMap, error::Error, fmt};

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

const DEFAULT_APP_ERROR_MSG: &str = "应用发生错误请稍后尝试";
const DEFAULT_NOT_FOUND_MSG: &str = "资源未找到，请核实后尝试";
const DEFAULT_BAD_REQUEST_MSG: &str = "客户端请求错误，请检查后重试";
const DEFAULT_SERVER_ERROR_MSG: &str = "服务器出现异常请稍后尝试";
const DEFAULT_SUCCESS_MSG: &str = "success";
const APP_ERROR_LOG_MSG: &str = "沙箱发生错误";
const PANIC_LOG_MSG: &str = "沙箱服务发生未定义异常";

/// 错误响应附加数据 / extra error response data.
pub type ErrorData = BTreeMap<String, String>;

/// 应用基础异常 / base application error.
#[derive(Debug, Clone)]
pub struct AppException<T = ErrorData> {
    pub msg: String,
    pub status_code: StatusCode,
    pub data: Option<T>,
}

/// 统一 API 响应体 / unified API response body.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ApiResponse<T = ErrorData> {
    pub code: u16,
    pub msg: String,
    pub data: T,
}

impl<T> ApiResponse<T> {
    pub fn success(data: Option<T>, msg: impl Into<String>) -> Self
    where
        T: Default,
    {
        Self {
            code: StatusCode::OK.as_u16(),
            msg: msg.into(),
            data: data.unwrap_or_default(),
        }
    }

    pub fn success_default(data: Option<T>) -> Self
    where
        T: Default,
    {
        Self::success(data, DEFAULT_SUCCESS_MSG)
    }

    /// 构造已接受并开始异步处理的响应。
    pub fn accepted(data: Option<T>, msg: impl Into<String>) -> Self
    where
        T: Default,
    {
        Self {
            code: StatusCode::ACCEPTED.as_u16(),
            msg: msg.into(),
            data: data.unwrap_or_default(),
        }
    }

    pub fn fail(code: StatusCode, msg: impl Into<String>, data: Option<T>) -> Self
    where
        T: Default,
    {
        Self {
            code: code.as_u16(),
            msg: msg.into(),
            data: data.unwrap_or_default(),
        }
    }
}

impl<T> Default for AppException<T>
where
    T: Default,
{
    fn default() -> Self {
        Self::new(
            DEFAULT_APP_ERROR_MSG,
            StatusCode::INTERNAL_SERVER_ERROR,
            None,
        )
    }
}

impl<T> AppException<T> {
    pub fn new(msg: impl Into<String>, status_code: StatusCode, data: Option<T>) -> Self {
        let msg = msg.into();
        tracing::error!(
            status_code = status_code.as_u16(),
            msg = %msg,
            "{}",
            APP_ERROR_LOG_MSG
        );

        Self {
            msg,
            status_code,
            data,
        }
    }
}

impl AppException {
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::new(msg, StatusCode::INTERNAL_SERVER_ERROR, None)
    }

    pub fn internal_default() -> Self {
        Self::internal(DEFAULT_SERVER_ERROR_MSG)
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::new(msg, StatusCode::NOT_FOUND, None)
    }

    pub fn not_found_default() -> Self {
        Self::not_found(DEFAULT_NOT_FOUND_MSG)
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::new(msg, StatusCode::BAD_REQUEST, None)
    }

    pub fn bad_request_default() -> Self {
        Self::bad_request(DEFAULT_BAD_REQUEST_MSG)
    }
}

impl<T> fmt::Display for AppException<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

impl<T> Error for AppException<T> where T: fmt::Debug {}

impl<T> From<AppException<T>> for ApiResponse<T>
where
    T: Default,
{
    fn from(err: AppException<T>) -> Self {
        Self::fail(err.status_code, err.msg, err.data)
    }
}

impl<T> IntoResponse for ApiResponse<T>
where
    T: Serialize,
{
    fn into_response(self) -> Response {
        Json(self).into_response()
    }
}

impl<T> IntoResponse for AppException<T>
where
    T: Serialize + Default,
{
    fn into_response(self) -> Response {
        let status_code = self.status_code;
        let body = ApiResponse::from(self);

        (status_code, Json(body)).into_response()
    }
}

pub async fn not_found() -> AppException {
    AppException::not_found_default()
}

pub fn handle_panic(err: Box<dyn Any + Send + 'static>) -> Response {
    if let Some(msg) = err.downcast_ref::<String>() {
        tracing::error!(panic = %msg, "{}", PANIC_LOG_MSG);
    } else if let Some(msg) = err.downcast_ref::<&str>() {
        tracing::error!(panic = %msg, "{}", PANIC_LOG_MSG);
    } else {
        tracing::error!("{}", PANIC_LOG_MSG);
    }

    AppException::internal_default().into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_exception_matches_python_internal_error() {
        let err: AppException = AppException::default();

        assert_eq!(err.status_code, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.msg, DEFAULT_APP_ERROR_MSG);
        assert_eq!(err.data, None);
    }

    #[test]
    fn constructors_match_python_specialized_errors() {
        let not_found = AppException::not_found_default();
        let bad_request = AppException::bad_request_default();

        assert_eq!(not_found.status_code, StatusCode::NOT_FOUND);
        assert_eq!(not_found.msg, DEFAULT_NOT_FOUND_MSG);
        assert_eq!(not_found.data, None);
        assert_eq!(bad_request.status_code, StatusCode::BAD_REQUEST);
        assert_eq!(bad_request.msg, DEFAULT_BAD_REQUEST_MSG);
        assert_eq!(bad_request.data, None);
    }

    #[test]
    fn app_exception_carries_optional_response_data() {
        let data = ErrorData::from([("field".to_string(), "name".to_string())]);
        let body = ApiResponse::from(AppException::new(
            "参数错误",
            StatusCode::BAD_REQUEST,
            Some(data),
        ));

        assert_eq!(body.code, StatusCode::BAD_REQUEST.as_u16());
        assert_eq!(body.msg, "参数错误");
        assert_eq!(body.data.get("field").map(String::as_str), Some("name"));
    }

    #[test]
    fn api_response_success_matches_python_base_response() {
        let body = ApiResponse::<ErrorData>::success_default(None);

        assert_eq!(body.code, StatusCode::OK.as_u16());
        assert_eq!(body.msg, DEFAULT_SUCCESS_MSG);
        assert!(body.data.is_empty());
    }

    #[test]
    fn api_response_accepted_marks_asynchronous_work_as_202() {
        let body = ApiResponse::<ErrorData>::accepted(None, "Supervisor重启任务已提交");

        assert_eq!(body.code, StatusCode::ACCEPTED.as_u16());
        assert_eq!(body.msg, "Supervisor重启任务已提交");
        assert!(body.data.is_empty());
    }

    #[test]
    fn api_response_fail_uses_empty_data_when_exception_has_no_data() {
        let body = ApiResponse::from(AppException::not_found_default());

        assert_eq!(body.code, StatusCode::NOT_FOUND.as_u16());
        assert_eq!(body.msg, DEFAULT_NOT_FOUND_MSG);
        assert!(body.data.is_empty());
    }

    #[test]
    fn into_response_sets_http_status_code() {
        let response = AppException::bad_request_default().into_response();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn not_found_returns_unified_json_response() {
        let response = not_found().await.into_response();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("not found response body should be readable");
        let body = String::from_utf8(body.to_vec()).expect("not found response should be utf-8");

        assert_eq!(
            body,
            r#"{"code":404,"msg":"资源未找到，请核实后尝试","data":{}}"#
        );
    }

    #[tokio::test]
    async fn handle_panic_hides_internal_error_detail() {
        let response = handle_panic(Box::new("database password leaked in panic"));

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("panic response body should be readable");
        let body = String::from_utf8(body.to_vec()).expect("panic response should be utf-8");

        assert_eq!(
            body,
            r#"{"code":500,"msg":"服务器出现异常请稍后尝试","data":{}}"#
        );
        assert!(!body.contains("database password"));
    }
}
