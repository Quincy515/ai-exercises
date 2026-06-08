use anyhow::Result;
use validator::Validate;

use crate::domain::models::{AgentConfig, AppConfig, LlmConfig, McpConfig};
use crate::domain::repositories::AppConfigRepository;

#[derive(Debug)]
pub struct McpServerNotFound {
    server_name: String,
}

impl McpServerNotFound {
    fn new(server_name: impl Into<String>) -> Self {
        Self {
            server_name: server_name.into(),
        }
    }
}

impl std::fmt::Display for McpServerNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "该MCP服务[{}]不存在，请核实后重试", self.server_name)
    }
}

impl std::error::Error for McpServerNotFound {}

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
            .trim()
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

    /// 获取 Agent 通用配置。
    /// Get Agent config.
    pub async fn get_agent_config(&self) -> Result<AgentConfig> {
        Ok(self.load_app_config().await?.agent_config)
    }

    /// 根据传递的 agent_config 更新 Agent 通用配置。
    pub async fn update_agent_config(&self, agent_config: AgentConfig) -> Result<AgentConfig> {
        // 1. 获取应用配置。
        // 1. Load application config.
        let mut config = self.load_app_config().await?;
        // 2. 更新 Agent 通用配置。
        // 2. Update Agent config.
        agent_config.validate()?;
        config.agent_config = agent_config;
        // 3. 调用仓库更新 app_config。
        // 3. Save app_config through the repository.
        self.app_config_repository.save(config.clone()).await?;
        Ok(config.agent_config)
    }

    /// 获取 MCP 配置。
    pub async fn get_mcp_config(&self) -> Result<McpConfig> {
        Ok(self.load_app_config().await?.mcp_config)
    }

    /// 根据传递的数据新增或更新 MCP 配置。
    pub async fn update_and_create_mcp_servers(&self, mcp_config: McpConfig) -> Result<McpConfig> {
        // 1. 获取应用配置
        // 1. Load application config
        let mut config = self.load_app_config().await?;

        // 2. 使用新的 mcp_config 更新原始的配置
        // 2. Update the original config with the new mcp_config.
        config.mcp_config.mcp_servers.extend(mcp_config.mcp_servers);
        config.mcp_config.validate()?;

        // 3. 调用数据仓库完成存储或更新
        self.app_config_repository.save(config.clone()).await?;
        Ok(config.mcp_config)
    }

    /// 根据传递的名字删除 MCP 服务
    pub async fn delete_mcp_server(&self, server_name: &str) -> Result<McpConfig> {
        // 1. 获取应用配置
        // 1. Load application config
        let mut config = self.load_app_config().await?;

        // 2. 从配置中删除指定的 MCP 服务器
        // 2. Remove the specified MCP server from the config.
        if config.mcp_config.mcp_servers.remove(server_name).is_none() {
            return Err(McpServerNotFound::new(server_name).into());
        }

        // 3. 调用数据仓库完成存储或更新
        self.app_config_repository.save(config.clone()).await?;
        Ok(config.mcp_config)
    }

    /// 更新MCP服务的启用状态
    pub async fn set_mcp_server_enabled(
        &self,
        server_name: &str,
        enabled: bool,
    ) -> Result<McpConfig> {
        // 1. 获取应用配置
        // 1. Load application config
        let mut config = self.load_app_config().await?;

        // 2. 查询对应服务的名字是否存在
        if let Some(server) = config.mcp_config.mcp_servers.get_mut(server_name) {
            // 3.如果存在则更新该MCP服务的启用状态
            server.enabled = enabled;
        } else {
            return Err(McpServerNotFound::new(server_name).into());
        }

        // 4. 调用数据仓库完成存储或更新
        // 4. Save the updated config to the repository.
        self.app_config_repository.save(config.clone()).await?;
        Ok(config.mcp_config)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use anyhow::Result;
    use async_trait::async_trait;

    use super::{AppConfigService, McpServerNotFound};
    use crate::domain::{
        models::{AgentConfig, AppConfig, LlmConfig, McpConfig, McpServerConfig},
        repositories::AppConfigRepository,
    };

    #[derive(Clone)]
    struct MemoryAppConfigRepository {
        config: Arc<Mutex<Option<AppConfig>>>,
    }

    impl MemoryAppConfigRepository {
        fn new(config: AppConfig) -> Self {
            Self {
                config: Arc::new(Mutex::new(Some(config))),
            }
        }
    }

    #[async_trait]
    impl AppConfigRepository for MemoryAppConfigRepository {
        async fn load(&self) -> Result<Option<AppConfig>> {
            Ok(self.config.lock().unwrap().clone())
        }

        async fn save(&self, config: AppConfig) -> Result<()> {
            *self.config.lock().unwrap() = Some(config);
            Ok(())
        }
    }

    #[tokio::test]
    async fn keeps_existing_api_key_when_update_contains_only_whitespace() {
        let repository = MemoryAppConfigRepository::new(AppConfig {
            llm_config: LlmConfig {
                api_key: Some("secret".to_string()),
                ..LlmConfig::default()
            },
            ..AppConfig::default()
        });
        let service = AppConfigService::new(repository);

        let updated = service
            .update_llm_config(LlmConfig {
                api_key: Some("  ".to_string()),
                ..LlmConfig::default()
            })
            .await
            .unwrap();

        assert_eq!(updated.api_key.as_deref(), Some("secret"));
    }

    #[tokio::test]
    async fn updates_agent_config() {
        let service = AppConfigService::new(MemoryAppConfigRepository::new(AppConfig::default()));
        let agent_config = AgentConfig {
            max_iterations: 200,
            max_retries: 4,
            max_search_results: 20,
        };

        let updated = service
            .update_agent_config(agent_config.clone())
            .await
            .unwrap();

        assert_eq!(updated, agent_config);
        assert_eq!(service.get_agent_config().await.unwrap(), agent_config);
    }

    #[tokio::test]
    async fn rejects_agent_config_outside_python_boundaries() {
        let service = AppConfigService::new(MemoryAppConfigRepository::new(AppConfig::default()));

        let result = service
            .update_agent_config(AgentConfig {
                max_iterations: 1000,
                ..AgentConfig::default()
            })
            .await;

        assert!(result.is_err());
        assert_eq!(
            service.get_agent_config().await.unwrap(),
            AgentConfig::default()
        );
    }

    #[tokio::test]
    async fn merges_mcp_servers_by_name() {
        let mut initial_mcp_config = McpConfig::default();
        initial_mcp_config.mcp_servers.insert(
            "existing".to_string(),
            streamable_http_server("https://old.example.com", true),
        );

        let service = AppConfigService::new(MemoryAppConfigRepository::new(AppConfig {
            mcp_config: initial_mcp_config,
            ..AppConfig::default()
        }));

        let mut incoming_mcp_config = McpConfig::default();
        incoming_mcp_config.mcp_servers.insert(
            "existing".to_string(),
            streamable_http_server("https://new.example.com", false),
        );
        incoming_mcp_config.mcp_servers.insert(
            "new".to_string(),
            streamable_http_server("https://mcp.example.com", true),
        );

        let updated = service
            .update_and_create_mcp_servers(incoming_mcp_config)
            .await
            .unwrap();

        assert_eq!(updated.mcp_servers.len(), 2);
        assert_eq!(
            updated.mcp_servers.get("existing").unwrap().url.as_deref(),
            Some("https://new.example.com")
        );
        assert!(!updated.mcp_servers.get("existing").unwrap().enabled);
        assert!(updated.mcp_servers.contains_key("new"));
    }

    #[tokio::test]
    async fn deletes_mcp_server_by_name() {
        let mut mcp_config = McpConfig::default();
        mcp_config.mcp_servers.insert(
            "demo".to_string(),
            streamable_http_server("https://mcp.example.com", true),
        );

        let service = AppConfigService::new(MemoryAppConfigRepository::new(AppConfig {
            mcp_config,
            ..AppConfig::default()
        }));

        let updated = service.delete_mcp_server("demo").await.unwrap();

        assert!(updated.mcp_servers.is_empty());
    }

    #[tokio::test]
    async fn reports_missing_mcp_server() {
        let service = AppConfigService::new(MemoryAppConfigRepository::new(AppConfig::default()));

        let err = service.delete_mcp_server("missing").await.unwrap_err();

        assert!(err.is::<McpServerNotFound>());
    }

    #[tokio::test]
    async fn sets_mcp_server_enabled() {
        let mut mcp_config = McpConfig::default();
        mcp_config.mcp_servers.insert(
            "demo".to_string(),
            streamable_http_server("https://mcp.example.com", false),
        );

        let service = AppConfigService::new(MemoryAppConfigRepository::new(AppConfig {
            mcp_config,
            ..AppConfig::default()
        }));

        let updated = service.set_mcp_server_enabled("demo", true).await.unwrap();

        assert!(updated.mcp_servers.get("demo").unwrap().enabled);
    }

    fn streamable_http_server(url: &str, enabled: bool) -> McpServerConfig {
        McpServerConfig {
            enabled,
            url: Some(url.to_string()),
            ..McpServerConfig::default()
        }
    }
}
