use anyhow::Result;
use async_trait::async_trait;

use crate::domain::models::AppConfig;

/// 应用配置仓库。
/// Repository interface for application config.
#[async_trait]
pub trait AppConfigRepository: Send + Sync {
    /// 加载获取应用配置。
    /// Load application config.
    async fn load(&self) -> Result<Option<AppConfig>>;

    /// 存储更新的应用配置。
    /// Save application config.
    async fn save(&self, config: AppConfig) -> Result<()>;
}
