use anyhow::Result;
use serde_json::{Map, Value};
use validator::Validate;

use crate::domain::models::{
    A2aConfig, A2aServerConfig, AgentConfig, AppConfig, LlmConfig, McpConfig, McpTransport,
};
use crate::domain::repositories::AppConfigRepository;
use crate::domain::services::tools::{a2a::A2AClientManager, McpClientManager};

/// MCP 服务器及其工具信息。
/// MCP server information with cached tool names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpServerToolInfo {
    pub server_name: String,
    pub enabled: bool,
    pub transport: McpTransport,
    pub tools: Vec<String>,
}

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

    /// 获取 MCP 服务器列表。
    pub async fn get_mcp_servers(&self) -> Result<Vec<McpServerToolInfo>> {
        // 1. 获取当前应用配置
        let app_config = self.load_app_config().await?;

        // 2. 创建 MCP 客户端管理器，对配置信息不进行过滤
        let mut mcp_client_manager = McpClientManager::new(Some(app_config.mcp_config.clone()));

        // 3. 初始化 MCP 客户端管理器
        mcp_client_manager.initialize().await;

        // 4. 获取 MCP 客户端管理器的工具列表
        let tools = mcp_client_manager.tools();

        // 5. 循环组装响应的工具格式
        let mut mcp_servers = app_config
            .mcp_config
            .mcp_servers
            .into_iter()
            .map(|(server_name, server_config)| McpServerToolInfo {
                tools: tools
                    .get(&server_name)
                    .map(|tools| {
                        tools
                            .iter()
                            .map(|tool| tool.name.to_string())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
                server_name,
                enabled: server_config.enabled,
                transport: server_config.transport,
            })
            .collect::<Vec<_>>();
        mcp_servers.sort_by(|left, right| left.server_name.cmp(&right.server_name));

        // 6. 清除 MCP 客户端管理器的相关资源
        mcp_client_manager.cleanup().await;

        Ok(mcp_servers)
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

    /// 根据传递的配置新增 A2A 服务器。
    pub async fn create_a2a_server(&self, base_url: &str) -> Result<A2aConfig> {
        // 1. 获取当前的应用配置
        let mut config = self.load_app_config().await?;

        // 2. 往数据中新增 A2A 服务（在新增之前其实可以检测下当前 Agent 是否存在）
        let a2a_server_config = A2aServerConfig::new(base_url);
        config.a2a_config.a2a_servers.push(a2a_server_config);
        config.a2a_config.validate()?;

        // 3. 调用数据仓库更新
        self.app_config_repository.save(config.clone()).await?;
        Ok(config.a2a_config)
    }

    /// 获取 A2A 服务列表。
    pub async fn get_a2a_servers(&self) -> Result<Vec<A2aServerAgentInfo>> {
        // 1. 获取当前的应用配置
        let app_config = self.load_app_config().await?;

        // 2. 构建 A2A 客户端管理器，对配置信息不过滤
        let mut a2a_client_manager = A2AClientManager::new(Some(app_config.a2a_config.clone()));

        let result: Result<Vec<A2aServerAgentInfo>> = async {
            // 3. 初始化 A2A 客户端管理器
            a2a_client_manager.initialize().await?;

            // 4. 获取 Agent 卡片列表
            let agent_cards = a2a_client_manager.agent_cards();

            // 5. 组装响应结构
            let mut a2a_servers = agent_cards
                .iter()
                .map(|(id, agent_card)| A2aServerAgentInfo::from_agent_card(id.clone(), agent_card))
                .collect::<Vec<_>>();
            a2a_servers.sort_by(|left, right| left.id.cmp(&right.id));

            Ok(a2a_servers)
        }
        .await;

        // 6. 清除客户端管理器资源
        a2a_client_manager.cleanup().await;
        result
    }

    /// 根据传递的 id + enabled 更新服务启用状态。
    pub async fn set_a2a_server_enabled(&self, a2a_id: &str, enabled: bool) -> Result<A2aConfig> {
        // 1. 获取当前的应用配置
        let mut config = self.load_app_config().await?;

        // 2. 查找需要更新的 A2A 服务并判断是否存在
        let server = config
            .a2a_config
            .a2a_servers
            .iter_mut()
            .find(|server| server.id == a2a_id)
            .ok_or_else(|| A2aServerNotFound::new(a2a_id))?;

        // 3. 如果存在则更新数据
        server.enabled = enabled;

        // 4. 调用数据仓库更新
        self.app_config_repository.save(config.clone()).await?;
        Ok(config.a2a_config)
    }

    /// 根据传递的 id 删除指定的 A2A 服务。
    pub async fn delete_a2a_server(&self, a2a_id: &str) -> Result<A2aConfig> {
        // 1. 获取当前的应用配置
        let mut config = self.load_app_config().await?;

        // 2. 计算需要删除位置的索引并判断是否存在
        let index = config
            .a2a_config
            .a2a_servers
            .iter()
            .position(|server| server.id == a2a_id)
            .ok_or_else(|| A2aServerNotFound::new(a2a_id))?;

        // 3. 删除 A2A 服务器
        config.a2a_config.a2a_servers.remove(index);

        // 4. 调用数据仓库更新
        self.app_config_repository.save(config.clone()).await?;
        Ok(config.a2a_config)
    }
}

#[derive(Debug)]
pub struct A2aServerNotFound {
    a2a_id: String,
}

impl A2aServerNotFound {
    fn new(a2a_id: impl Into<String>) -> Self {
        Self {
            a2a_id: a2a_id.into(),
        }
    }
}

impl std::fmt::Display for A2aServerNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "该A2A服务[{}]不存在，请核实后重试", self.a2a_id)
    }
}

impl std::error::Error for A2aServerNotFound {}

/// A2A 服务器及其远程 Agent 卡片信息。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct A2aServerAgentInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub input_modes: Vec<String>,
    pub output_modes: Vec<String>,
    pub streaming: bool,
    pub push_notifications: bool,
    pub enabled: bool,
}

impl A2aServerAgentInfo {
    fn from_agent_card(id: String, agent_card: &Map<String, Value>) -> Self {
        let capabilities = agent_card.get("capabilities").and_then(Value::as_object);

        Self {
            id,
            name: string_field(agent_card, "name"),
            description: string_field(agent_card, "description"),
            input_modes: string_list_field(agent_card, "defaultInputModes"),
            output_modes: string_list_field(agent_card, "defaultOutputModes"),
            streaming: capabilities
                .and_then(|value| value.get("streaming"))
                .and_then(Value::as_bool)
                .unwrap_or_default(),
            push_notifications: capabilities
                .and_then(|value| value.get("push_notifications"))
                .and_then(Value::as_bool)
                .unwrap_or_default(),
            enabled: agent_card
                .get("enabled")
                .and_then(Value::as_bool)
                .unwrap_or_default(),
        }
    }
}

fn string_field(agent_card: &Map<String, Value>, field: &str) -> String {
    agent_card
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn string_list_field(agent_card: &Map<String, Value>, field: &str) -> Vec<String> {
    agent_card
        .get(field)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::{
        net::Ipv4Addr,
        sync::{Arc, Mutex},
    };

    use anyhow::Result;
    use async_trait::async_trait;
    use axum::{routing::get, Json, Router};
    use tokio::net::TcpListener;

    use super::{A2aServerAgentInfo, A2aServerNotFound, AppConfigService, McpServerNotFound};
    use crate::domain::{
        models::{
            A2aConfig, A2aServerConfig, AgentConfig, AppConfig, LlmConfig, McpConfig,
            McpServerConfig, McpTransport,
        },
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
    async fn lists_disabled_mcp_servers_without_tools() {
        let mut mcp_config = McpConfig::default();
        mcp_config.mcp_servers.insert(
            "disabled".to_string(),
            McpServerConfig {
                transport: McpTransport::Stdio,
                enabled: false,
                command: Some("unused".to_string()),
                ..McpServerConfig::default()
            },
        );
        let service = AppConfigService::new(MemoryAppConfigRepository::new(AppConfig {
            mcp_config,
            ..AppConfig::default()
        }));

        let servers = service.get_mcp_servers().await.unwrap();

        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].server_name, "disabled");
        assert!(!servers[0].enabled);
        assert_eq!(servers[0].transport, McpTransport::Stdio);
        assert!(servers[0].tools.is_empty());
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

    #[tokio::test]
    async fn creates_enabled_a2a_server_and_saves_it() {
        let service = AppConfigService::new(MemoryAppConfigRepository::new(AppConfig::default()));

        let updated = service
            .create_a2a_server("http://localhost:9999")
            .await
            .unwrap();

        assert_eq!(updated.a2a_servers.len(), 1);
        assert_eq!(updated.a2a_servers[0].base_url, "http://localhost:9999");
        assert!(updated.a2a_servers[0].enabled);
        assert_eq!(service.load_app_config().await.unwrap().a2a_config, updated);
    }

    #[tokio::test]
    async fn combines_agent_card_with_local_a2a_enabled_state() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/.well-known/agent-card.json",
            get(|| async {
                Json(serde_json::json!({
                    "name": "Writer Agent",
                    "description": "撰写文章",
                    "enabled": true,
                    "defaultInputModes": ["text"],
                    "defaultOutputModes": ["text", "file"],
                    "capabilities": {
                        "streaming": true,
                        "push_notifications": true
                    }
                }))
            }),
        );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let service = AppConfigService::new(MemoryAppConfigRepository::new(AppConfig {
            a2a_config: A2aConfig {
                a2a_servers: vec![A2aServerConfig {
                    id: "writer-agent".to_string(),
                    base_url: format!("http://{address}"),
                    enabled: false,
                }],
            },
            ..AppConfig::default()
        }));

        let servers = service.get_a2a_servers().await.unwrap();
        server.abort();

        assert_eq!(
            servers,
            vec![A2aServerAgentInfo {
                id: "writer-agent".to_string(),
                name: "Writer Agent".to_string(),
                description: "撰写文章".to_string(),
                input_modes: vec!["text".to_string()],
                output_modes: vec!["text".to_string(), "file".to_string()],
                streaming: true,
                push_notifications: true,
                enabled: false,
            }]
        );
    }

    #[tokio::test]
    async fn updates_and_deletes_a2a_server_by_id() {
        let service = AppConfigService::new(MemoryAppConfigRepository::new(AppConfig {
            a2a_config: A2aConfig {
                a2a_servers: vec![A2aServerConfig {
                    id: "writer-agent".to_string(),
                    base_url: "http://localhost:9999".to_string(),
                    enabled: true,
                }],
            },
            ..AppConfig::default()
        }));

        let updated = service
            .set_a2a_server_enabled("writer-agent", false)
            .await
            .unwrap();
        assert!(!updated.a2a_servers[0].enabled);

        let updated = service.delete_a2a_server("writer-agent").await.unwrap();
        assert!(updated.a2a_servers.is_empty());

        let err = service
            .set_a2a_server_enabled("writer-agent", true)
            .await
            .unwrap_err();
        assert!(err.is::<A2aServerNotFound>());

        let err = service.delete_a2a_server("writer-agent").await.unwrap_err();
        assert!(err.is::<A2aServerNotFound>());
    }

    fn streamable_http_server(url: &str, enabled: bool) -> McpServerConfig {
        McpServerConfig {
            enabled,
            url: Some(url.to_string()),
            ..McpServerConfig::default()
        }
    }
}
