use serde::Deserialize;
use utoipa::ToSchema;

/// 激活超时销毁请求
#[derive(Debug, Deserialize, ToSchema, PartialEq, Eq)]
pub struct TimeoutRequest {
    /// 分钟数
    pub minutes: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_request_accepts_explicit_or_omitted_minutes() {
        let explicit = TimeoutRequest { minutes: Some(10) };
        let omitted = TimeoutRequest { minutes: None };

        assert_eq!(explicit.minutes, Some(10));
        assert_eq!(omitted.minutes, None);
    }
}
