use anyhow::{Ok, Result};

use crate::domain::models::{AppConfig, LlmConfig};
use crate::domain::repositories::AppConfigRepository;

/// 应用配置服务
pub struct AppConfigService<R: AppConfigRepository> {
    app_config_repository: R,
}

impl<R: AppConfigRepository> AppConfigService<R> {
    pub fn new(app_config_repository: R) -> Self {
        Self {
            app_config_repository,
        }
    }

    /// 加载获取所有的应用配置
    fn load_app_config(&self) -> AppConfig {
        self.app_config_repository.load().unwrap_or_default()
    }

    /// 获取 LLM 提供商配置
    pub fn get_llm_config(&self) -> Result<LlmConfig> {
        Ok(self.load_app_config().llm_config)
    }

    /// 根据传递的 llm_config 更新语言模型提供商配置
    pub fn update_llm_config(&self, llm_config: LlmConfig) -> Result<LlmConfig> {
        // 1. 获取应用配置
        let mut config = self.load_app_config();
        let old_api_key = config.llm_config.api_key.clone();
        let mut next_llm_config = llm_config;

        // 2. 判断 api_key 是否为空
        if next_llm_config
            .api_key
            .as_deref()
            .unwrap_or_default()
            .is_empty()
        {
            next_llm_config.api_key = old_api_key;
        };

        // 3. 调用函数更新 app_config
        config.llm_config = next_llm_config;
        self.app_config_repository.save(config.clone())?;

        Ok(config.llm_config)
    }
}
