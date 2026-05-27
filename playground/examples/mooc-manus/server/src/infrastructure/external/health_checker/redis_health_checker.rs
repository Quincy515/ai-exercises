use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use loco_rs::cache::Cache;
use tracing::error;

use crate::domain::{external::HealthChecker, models::HealthStatus};

/// Redis 健康检查器
/// Redis health checker.
pub struct RedisHealthChecker {
    cache: Arc<Cache>,
}

impl RedisHealthChecker {
    pub fn new(cache: Arc<Cache>) -> Self {
        Self { cache }
    }
}

#[async_trait]
impl HealthChecker for RedisHealthChecker {
    /// 调用 Loco cache 的 ping 判断 Redis 服务是否正常。
    /// Use the Loco cache ping to check whether Redis is healthy.
    async fn check(&self) -> Result<HealthStatus> {
        match self.cache.ping().await {
            Ok(()) => Ok(HealthStatus {
                service: Some("redis".to_string()),
                status: Some("ok".to_string()),
                details: None,
            }),
            Err(err) => {
                error!(error = %err, "Redis 健康检查失败");
                Ok(HealthStatus {
                    service: Some("redis".to_string()),
                    status: Some("error".to_string()),
                    details: Some(err.to_string()),
                })
            }
        }
    }
}
