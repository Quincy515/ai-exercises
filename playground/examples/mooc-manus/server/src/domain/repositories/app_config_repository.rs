use anyhow::Result;

use crate::domain::models::AppConfig;

/// 应用配置仓库
pub trait AppConfigRepository {
    /// 加载获取应用配置
    fn load(&self) -> Option<AppConfig>;

    /// 存储更新的应用配置
    fn save(&self, config: AppConfig) -> Result<()>;
}
