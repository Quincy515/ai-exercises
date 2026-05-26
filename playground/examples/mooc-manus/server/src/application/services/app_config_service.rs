use anyhow::Result;
use validator::Validate;

use crate::domain::models::{AppConfig, LlmConfig};
use crate::domain::repositories::AppConfigRepository;

/// 应用配置服务。
/// Application config service.
pub struct AppConfigService<R: AppConfigRepository> {
    app_config_repository: R,
}

impl<R: AppConfigRepository> AppConfigService<R> {
    pub fn new(app_config_repository: R) -> Self {
        Self {
            app_config_repository,
        }
    }

    /// 加载获取所有的应用配置。
    /// Load the full application config.
    async fn load_app_config(&self) -> Result<AppConfig> {
        Ok(self.app_config_repository.load().await?.unwrap_or_default())
    }

    /// 获取 LLM 提供商配置。
    /// Get LLM provider config.
    pub async fn get_llm_config(&self) -> Result<LlmConfig> {
        Ok(self.load_app_config().await?.llm_config)
    }

    /// 根据传递的 llm_config 更新语言模型提供商配置。
    /// Update LLM provider config from the given config.
    pub async fn update_llm_config(&self, llm_config: LlmConfig) -> Result<LlmConfig> {
        // 1. 获取应用配置。
        // 1. Load application config.
        let mut config = self.load_app_config().await?;
        let old_api_key = config.llm_config.api_key.clone();
        let mut next_llm_config = llm_config;

        // 2. 判断 api_key 是否为空。
        // 2. Keep the old api_key when the incoming api_key is empty.
        if next_llm_config
            .api_key
            .as_deref()
            .unwrap_or_default()
            .is_empty()
        {
            next_llm_config.api_key = old_api_key;
        }
        next_llm_config.validate()?;

        // 3. 调用仓库更新 app_config。
        // 3. Save app_config through the repository.
        config.llm_config = next_llm_config;
        self.app_config_repository.save(config.clone()).await?;

        Ok(config.llm_config)
    }
}
