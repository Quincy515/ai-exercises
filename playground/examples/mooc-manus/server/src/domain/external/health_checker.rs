use anyhow::Result;
use async_trait::async_trait;

use crate::domain::models::HealthStatus;

/// 服务健康检查协议
/// Service health check protocol.
#[async_trait]
pub trait HealthChecker: Send + Sync {
    /// 检查对应的服务是否健康
    /// Check whether the target service is healthy.
    async fn check(&self) -> Result<HealthStatus>;
}
