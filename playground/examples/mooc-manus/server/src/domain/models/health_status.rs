use serde::{Deserialize, Serialize};

/// 健康检查状态
/// Health check status.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct HealthStatus {
    /// 健康检查对应的服务名字
    /// Service name for this health check.
    pub service: Option<String>,
    /// 健康检查状态，支持 ok 表示正常，error 表示出错
    /// Health status. Use ok for healthy and error for failures.
    pub status: Option<String>,
    /// 出错时的详情提示
    /// Details for the health check result.
    pub details: Option<String>,
}
