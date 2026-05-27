use std::sync::Arc;

use crate::domain::{external::HealthChecker, models::HealthStatus};

/// 状态服务，用于检查系统的服务状态
/// Status service for checking system service health.
pub struct StatusService {
    checkers: Vec<Arc<dyn HealthChecker>>,
}

impl StatusService {
    /// 构造函数，传递所有检查器完成服务初始化
    /// Build the service from all health checkers.
    pub fn new(checkers: Vec<Arc<dyn HealthChecker>>) -> Self {
        Self { checkers }
    }

    /// 调用所有检查器发起检查并返回对应的健康状态
    /// Run all checkers and return their health statuses.
    pub async fn check_all(&self) -> Vec<HealthStatus> {
        // 1. 并行调用所有服务进行检查
        // 1. Run all service checks concurrently.
        let handles: Vec<_> = self
            .checkers
            .iter()
            .map(|checker| {
                let checker = Arc::clone(checker);
                tokio::spawn(async move { checker.check().await })
            })
            .collect();

        // 2. 处理可能发生的异常
        // 2. Convert checker failures into health status entries.
        let mut processed_results = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(Ok(status)) => processed_results.push(status),
                Ok(Err(err)) => processed_results.push(HealthStatus {
                    service: Some("未知服务".to_string()),
                    status: Some("error".to_string()),
                    details: Some(format!("未知检查器发生错误: {err}")),
                }),
                Err(err) => processed_results.push(HealthStatus {
                    service: Some("未知服务".to_string()),
                    status: Some("error".to_string()),
                    details: Some(format!("未知检查器发生错误: {err}")),
                }),
            }
        }

        processed_results
    }
}
