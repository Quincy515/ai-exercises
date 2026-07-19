use std::{borrow::Cow, collections::HashMap};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;
use validator::{Validate, ValidationError};

/// 语言模型配置
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Validate)]
pub struct LlmConfig {
    #[validate(url)]
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model_name: Option<String>,
    #[validate(range(min = -2.0, max = 2.0, message = "temperature 必须在 -2.0 到 2.0 之间"))]
    pub temperature: Option<f32>,
    #[validate(range(min = 0, message = "最大输出 tokens 数必须大于等于 0"))]
    pub max_tokens: Option<usize>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: Some("https://api.deepseek.com".to_string()),
            api_key: None,
            model_name: Some("deepseek-v4-pro".to_string()),
            temperature: Some(0.7),
            max_tokens: Some(8192),
        }
    }
}

/// Agent 通用配置
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Validate)]
pub struct AgentConfig {
    /// 最大迭代次数
    #[validate(range(min = 1, max = 999, message = "max_iterations 必须在 1 到 999 之间"))]
    pub max_iterations: usize,
    /// LLM / 工具最大重试次数
    #[validate(range(min = 2, max = 9, message = "max_retries 必须在 2 到 9 之间"))]
    pub max_retries: usize,
    /// 最大搜索结果数
    #[validate(range(min = 2, max = 29, message = "max_search_results 必须在 2 到 29 之间"))]
    pub max_search_results: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            max_retries: 3,
            max_search_results: 10,
        }
    }
}

/// MCP 传输类型枚举
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    /// 本地输入输出
    Stdio,
    /// 可流式的 HTTP
    #[default]
    StreamableHttp,
}

/// MCP 单条服务配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Validate)]
#[validate(schema(function = "validate_mcp_server_config"))]
#[serde(default)]
pub struct McpServerConfig {
    /// 传输协议
    pub transport: McpTransport,
    /// 是否开启
    pub enabled: bool,
    /// MCP 服务的描述
    pub description: Option<String>,
    /// 环境变量配置
    pub env: Option<Map<String, Value>>,
    /// stdio 启动命令
    pub command: Option<String>,
    /// stdio 命令参数
    pub args: Option<Vec<String>>,
    /// Streamable HTTP 服务 URL
    pub url: Option<String>,
    /// Streamable HTTP 请求头
    pub headers: Option<Map<String, Value>>,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            transport: McpTransport::StreamableHttp,
            enabled: true,
            description: None,
            env: None,
            command: None,
            args: None,
            url: None,
            headers: None,
        }
    }
}

/// 校验 mcp_server_config 的相关信息，包含 url+command
fn validate_mcp_server_config(config: &McpServerConfig) -> Result<(), ValidationError> {
    match config.transport {
        // 1. 判断 transport 是否为 streamable_http
        McpTransport::StreamableHttp
            // 2. streamable_http 需要判断 url 是否传递
            if config.url.as_deref().unwrap_or_default().is_empty() => {
            Err(ValidationError::new("mcp_server_url_required")
                .with_message(Cow::Borrowed("在streamable_http模式下必须传递url")))
        }
        // 3. 判断 transport 是否为 stdio 类型
        McpTransport::Stdio if config.command.as_deref().unwrap_or_default().is_empty() => {
            // 4. 判断 command 也就是启动命令是否传递
            Err(ValidationError::new("mcp_server_command_required")
                .with_message(Cow::Borrowed("在stdio模式下必须传递command")))
        }
        _ => Ok(()),
    }
}

/// 应用 MCP 配置
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Validate)]
#[serde(default)]
pub struct McpConfig {
    /// MCP 服务器配置
    #[serde(rename = "mcpServers", alias = "mcp_servers")]
    #[validate(nested)]
    pub mcp_servers: HashMap<String, McpServerConfig>,
}

/// A2A 服务配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Validate)]
pub struct A2aServerConfig {
    /// 唯一标识
    #[serde(default = "default_a2a_server_id")]
    pub id: String,
    /// 服务基础 URL
    pub base_url: String,
    /// 服务是否启用
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_a2a_server_id() -> String {
    Uuid::new_v4().to_string()
}

fn default_enabled() -> bool {
    true
}

/// A2A 配置
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Validate)]
#[serde(default)]
pub struct A2aConfig {
    /// A2A 服务配置
    #[validate(nested)]
    pub a2a_servers: Vec<A2aServerConfig>,
}

/// 应用配置信息，包含Agent配置、LLM提供商、A2A网络、MCP服务配置等
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default, Validate)]
pub struct AppConfig {
    /// 语言模型配置
    #[serde(default)]
    #[validate(nested)]
    pub llm_config: LlmConfig,
    /// Agent 通用配置
    #[serde(default)]
    #[validate(nested)]
    pub agent_config: AgentConfig,
    /// MCP 服务配置
    #[serde(default)]
    #[validate(nested)]
    pub mcp_config: McpConfig,
    /// A2A 配置
    #[serde(default)]
    #[validate(nested)]
    pub a2a_config: A2aConfig,
}

#[cfg(test)]
mod tests {
    use super::{
        A2aConfig, A2aServerConfig, AgentConfig, AppConfig, LlmConfig, McpConfig, McpServerConfig,
        McpTransport,
    };
    use uuid::Uuid;
    use validator::Validate;

    #[test]
    fn creates_python_like_defaults() {
        let config = AppConfig::default();

        assert_eq!(config.llm_config, LlmConfig::default());
        assert_eq!(
            config.llm_config.model_name.as_deref(),
            Some("deepseek-v4-pro")
        );
        assert_eq!(config.agent_config, AgentConfig::default());
        assert_eq!(config.mcp_config, McpConfig::default());
        assert_eq!(config.a2a_config, A2aConfig::default());
        assert_eq!(
            McpServerConfig::default().transport,
            McpTransport::StreamableHttp
        );
        assert!(McpServerConfig::default().enabled);
    }

    #[test]
    fn validates_agent_config_with_python_strict_boundaries() {
        assert!(AgentConfig {
            max_iterations: 1,
            max_retries: 2,
            max_search_results: 2,
        }
        .validate()
        .is_ok());

        for invalid_config in [
            AgentConfig {
                max_iterations: 0,
                ..AgentConfig::default()
            },
            AgentConfig {
                max_iterations: 1000,
                ..AgentConfig::default()
            },
            AgentConfig {
                max_retries: 1,
                ..AgentConfig::default()
            },
            AgentConfig {
                max_retries: 10,
                ..AgentConfig::default()
            },
            AgentConfig {
                max_search_results: 1,
                ..AgentConfig::default()
            },
            AgentConfig {
                max_search_results: 30,
                ..AgentConfig::default()
            },
        ] {
            assert!(invalid_config.validate().is_err());
        }
    }

    #[test]
    fn loads_legacy_config_without_agent_or_mcp_settings() {
        let config: AppConfig = serde_json::from_value(serde_json::json!({
            "llm_config": LlmConfig::default()
        }))
        .unwrap();

        assert_eq!(config.agent_config, AgentConfig::default());
        assert_eq!(config.mcp_config, McpConfig::default());
        assert_eq!(config.a2a_config, A2aConfig::default());
    }

    #[test]
    fn generates_unique_a2a_server_ids() {
        let first: A2aServerConfig = serde_json::from_value(serde_json::json!({
            "base_url": "https://first-agent.example.com"
        }))
        .unwrap();
        let second: A2aServerConfig = serde_json::from_value(serde_json::json!({
            "base_url": "https://second-agent.example.com"
        }))
        .unwrap();

        assert!(Uuid::parse_str(&first.id).is_ok());
        assert!(Uuid::parse_str(&second.id).is_ok());
        assert_ne!(first.id, second.id);
    }

    #[test]
    fn fills_a2a_id_and_enabled_when_deserializing() {
        let server: A2aServerConfig = serde_json::from_value(serde_json::json!({
            "base_url": "https://agent.example.com"
        }))
        .unwrap();

        assert!(Uuid::parse_str(&server.id).is_ok());
        assert_eq!(server.base_url, "https://agent.example.com");
        assert!(server.enabled);
    }

    #[test]
    fn requires_a2a_base_url_like_python_model() {
        let result = serde_json::from_value::<A2aServerConfig>(serde_json::json!({}));

        assert!(result.is_err());
    }

    #[test]
    fn validates_mcp_server_fields_for_each_transport() {
        assert!(McpServerConfig::default().validate().is_err());

        assert!(McpServerConfig {
            url: Some("https://mcp.example.com".to_string()),
            ..McpServerConfig::default()
        }
        .validate()
        .is_ok());

        assert!(McpServerConfig {
            transport: McpTransport::Stdio,
            command: Some("npx".to_string()),
            ..McpServerConfig::default()
        }
        .validate()
        .is_ok());

        let mut mcp_config = McpConfig::default();
        mcp_config
            .mcp_servers
            .insert("invalid".to_string(), McpServerConfig::default());
        assert!(AppConfig {
            mcp_config,
            ..AppConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn uses_python_mcp_json_shape() {
        let config: AppConfig = serde_json::from_value(serde_json::json!({
            "llm_config": LlmConfig::default(),
            "agent_config": AgentConfig::default(),
            "mcp_config": {
                "mcpServers": {
                    "demo": {
                        "transport": "streamable_http",
                        "url": "https://mcp.example.com",
                        "custom_server_field": "server-value"
                    }
                },
                "custom_mcp_field": "mcp-value"
            },
            "custom_app_field": "app-value"
        }))
        .unwrap();

        assert!(config.validate().is_ok());

        let serialized = serde_json::to_value(config).unwrap();
        assert_eq!(
            serialized["mcp_config"]["mcpServers"]["demo"]["transport"],
            "streamable_http"
        );
        assert!(serialized["mcp_config"].get("mcp_servers").is_none());
    }
}
