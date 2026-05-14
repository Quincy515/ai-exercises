use axum::http::StatusCode;
use loco_rs::{controller::ErrorDetail, Error};
use serde_json::Value;

#[derive(Debug)]
pub enum AppError {
    Business {
        status: StatusCode,
        code: &'static str,
        message: String,
        details: Option<Value>,
    },
    Internal {
        code: &'static str,
        message: String,
    },
}

impl AppError {
    #[must_use]
    pub fn business(
        status: StatusCode,
        code: &'static str,
        message: impl Into<String>,
        details: Option<Value>,
    ) -> Self {
        Self::Business {
            status,
            code,
            message: message.into(),
            details,
        }
    }

    #[must_use]
    pub fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self::business(StatusCode::BAD_REQUEST, code, message, None)
    }

    #[must_use]
    pub fn validation(code: &'static str, message: impl Into<String>, details: Value) -> Self {
        Self::business(
            StatusCode::UNPROCESSABLE_ENTITY,
            code,
            message,
            Some(details),
        )
    }

    #[must_use]
    pub fn too_many_requests(code: &'static str, message: impl Into<String>) -> Self {
        Self::business(StatusCode::TOO_MANY_REQUESTS, code, message, None)
    }

    #[must_use]
    pub fn internal(code: &'static str, message: impl Into<String>) -> Self {
        Self::Internal {
            code,
            message: message.into(),
        }
    }
}

impl From<AppError> for Error {
    fn from(err: AppError) -> Self {
        match err {
            AppError::Business {
                status,
                code,
                message,
                details,
            } => Error::CustomError(
                status,
                ErrorDetail {
                    error: Some(code.to_string()),
                    description: Some(message),
                    errors: details,
                },
            ),
            AppError::Internal { code, message } => {
                // 记录项目上下文，HTTP 出口日志继续交给 Loco。
                // Record application context, while Loco keeps the HTTP exit log.
                tracing::error!(code, message, "app_internal_error");

                Error::CustomError(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ErrorDetail::new("internal_server_error", "Internal Server Error"),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AppError;
    use axum::http::StatusCode;
    use loco_rs::Error;
    use serde_json::json;

    #[test]
    fn business_error_maps_to_custom_error() {
        let err: Error = AppError::bad_request("auth.invalid_state", "登录状态无效").into();

        let Error::CustomError(status, detail) = err else {
            panic!("expected custom error");
        };

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(detail.error.as_deref(), Some("auth.invalid_state"));
        assert_eq!(detail.description.as_deref(), Some("登录状态无效"));
        assert!(detail.errors.is_none());
    }

    #[test]
    fn validation_error_maps_to_422_custom_error() {
        let details = json!({
            "email": ["邮箱格式错误"],
            "password": ["密码长度不足"]
        });
        let err: Error = AppError::validation(
            "request.validation_failed",
            "请求参数校验失败",
            details.clone(),
        )
        .into();

        let Error::CustomError(status, detail) = err else {
            panic!("expected custom error");
        };

        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(detail.error.as_deref(), Some("request.validation_failed"));
        assert_eq!(detail.description.as_deref(), Some("请求参数校验失败"));
        assert_eq!(detail.errors, Some(details));
    }

    #[test]
    fn internal_error_maps_to_safe_500_custom_error() {
        let err: Error = AppError::internal("user.find_failed", "failed to find user").into();

        let Error::CustomError(status, detail) = err else {
            panic!("expected custom error");
        };

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(detail.error.as_deref(), Some("internal_server_error"));
        assert_eq!(detail.description.as_deref(), Some("Internal Server Error"));
        assert!(detail.errors.is_none());
    }
}
