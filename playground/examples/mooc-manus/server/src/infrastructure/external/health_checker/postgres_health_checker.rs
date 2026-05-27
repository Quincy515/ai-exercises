use async_trait::async_trait;
use sea_orm::{ConnectionTrait, DatabaseConnection, Statement};
use tracing::error;

use crate::domain::{external::HealthChecker, models::HealthStatus};

/// Postgres 健康检查器
/// Postgres health checker.
pub struct PostgresHealthChecker {
    db: DatabaseConnection,
}

impl PostgresHealthChecker {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

#[async_trait]
impl HealthChecker for PostgresHealthChecker {
    /// 执行一段简单 SQL 判断数据库服务是否正常。
    /// Run a simple SQL statement to check whether the database is healthy.
    async fn check(&self) -> anyhow::Result<HealthStatus> {
        let statement = Statement::from_string(self.db.get_database_backend(), "SELECT 1");

        match self.db.execute(statement).await {
            Ok(_) => Ok(HealthStatus {
                service: Some("postgres".to_string()),
                status: Some("ok".to_string()),
                details: None,
            }),
            Err(err) => {
                error!(error = %err, "Postgres 健康检查失败");
                Ok(HealthStatus {
                    service: Some("postgres".to_string()),
                    status: Some("error".to_string()),
                    details: Some(err.to_string()),
                })
            }
        }
    }
}
