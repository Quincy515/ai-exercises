use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use utoipa::ToSchema;

use crate::domain::models::{AgentConfig, LlmConfig, McpConfig, McpServerConfig, McpTransport};

/// LLM 配置更新请求。
/// Request payload for updating LLM config.
#[derive(Debug, Deserialize, ToSchema)]
pub struct LlmConfigRequest {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model_name: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

impl From<LlmConfigRequest> for LlmConfig {
    fn from(request: LlmConfigRequest) -> Self {
        Self {
            base_url: request.base_url,
            api_key: request.api_key,
            model_name: request.model_name,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
        }
    }
}

/// LLM 配置响应。
/// Response payload for LLM config.
#[derive(Debug, Serialize, ToSchema)]
pub struct LlmConfigResponse {
    pub base_url: Option<String>,
    pub api_key_configured: bool,
    pub model_name: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

impl From<LlmConfig> for LlmConfigResponse {
    fn from(config: LlmConfig) -> Self {
        Self {
            base_url: config.base_url,
            api_key_configured: config
                .api_key
                .as_deref()
                .is_some_and(|api_key| !api_key.is_empty()),
            model_name: config.model_name,
            temperature: config.temperature,
            max_tokens: config.max_tokens,
        }
    }
}

/// Agent 配置更新请求。
/// Request payload for updating Agent config.
#[derive(Debug, Deserialize, ToSchema)]
pub struct AgentConfigRequest {
    pub max_iterations: usize,
    pub max_retries: usize,
    pub max_search_results: usize,
}

impl From<AgentConfigRequest> for AgentConfig {
    fn from(request: AgentConfigRequest) -> Self {
        Self {
            max_iterations: request.max_iterations,
            max_retries: request.max_retries,
            max_search_results: request.max_search_results,
        }
    }
}

/// Agent 配置响应。
/// Response payload for Agent config.
#[derive(Debug, Serialize, ToSchema)]
pub struct AgentConfigResponse {
    pub max_iterations: usize,
    pub max_retries: usize,
    pub max_search_results: usize,
}

impl From<AgentConfig> for AgentConfigResponse {
    fn from(config: AgentConfig) -> Self {
        Self {
            max_iterations: config.max_iterations,
            max_retries: config.max_retries,
            max_search_results: config.max_search_results,
        }
    }
}

/// MCP 传输类型。
/// MCP transport type for HTTP payloads.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum McpTransportPayload {
    Stdio,
    Sse,
    #[default]
    StreamableHttp,
}

impl From<McpTransportPayload> for McpTransport {
    fn from(transport: McpTransportPayload) -> Self {
        match transport {
            McpTransportPayload::Stdio => Self::Stdio,
            McpTransportPayload::Sse => Self::Sse,
            McpTransportPayload::StreamableHttp => Self::StreamableHttp,
        }
    }
}

impl From<McpTransport> for McpTransportPayload {
    fn from(transport: McpTransport) -> Self {
        match transport {
            McpTransport::Stdio => Self::Stdio,
            McpTransport::Sse => Self::Sse,
            McpTransport::StreamableHttp => Self::StreamableHttp,
        }
    }
}

/// 单个 MCP 服务配置请求。
/// Request payload for one MCP server.
#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct McpServerConfigRequest {
    #[serde(default)]
    pub transport: McpTransportPayload,
    #[serde(default = "default_mcp_enabled")]
    pub enabled: bool,
    pub description: Option<String>,
    #[schema(value_type = Object)]
    pub env: Option<Map<String, Value>>,
    pub command: Option<String>,
    pub args: Option<Vec<String>>,
    pub url: Option<String>,
    #[schema(value_type = Object)]
    pub headers: Option<Map<String, Value>>,
}

impl From<McpServerConfigRequest> for McpServerConfig {
    fn from(request: McpServerConfigRequest) -> Self {
        Self {
            transport: request.transport.into(),
            enabled: request.enabled,
            description: request.description,
            env: request.env,
            command: request.command,
            args: request.args,
            url: request.url,
            headers: request.headers,
        }
    }
}

/// MCP 配置请求。
/// Request payload for updating MCP config.
#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct McpConfigRequest {
    #[serde(default, rename = "mcpServers", alias = "mcp_servers")]
    pub mcp_servers: BTreeMap<String, McpServerConfigRequest>,
}

impl From<McpConfigRequest> for McpConfig {
    fn from(request: McpConfigRequest) -> Self {
        Self {
            mcp_servers: request
                .mcp_servers
                .into_iter()
                .map(|(name, server)| (name, server.into()))
                .collect::<HashMap<_, _>>(),
        }
    }
}

/// 单个 MCP 服务配置响应。
/// Response payload for one MCP server.
#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct McpServerConfigResponse {
    pub transport: McpTransportPayload,
    pub enabled: bool,
    pub description: Option<String>,
    #[schema(value_type = Object)]
    pub env: Option<Value>,
    pub command: Option<String>,
    pub args: Option<Vec<String>>,
    pub url: Option<String>,
    #[schema(value_type = Object)]
    pub headers: Option<Value>,
}

impl From<McpServerConfig> for McpServerConfigResponse {
    fn from(config: McpServerConfig) -> Self {
        Self {
            transport: config.transport.into(),
            enabled: config.enabled,
            description: config.description,
            env: config.env.map(Value::Object),
            command: config.command,
            args: config.args,
            url: config.url,
            headers: config.headers.map(Value::Object),
        }
    }
}

/// MCP 配置响应。
/// Response payload for MCP config.
#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct McpConfigResponse {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: BTreeMap<String, McpServerConfigResponse>,
}

impl From<McpConfig> for McpConfigResponse {
    fn from(config: McpConfig) -> Self {
        Self {
            mcp_servers: config
                .mcp_servers
                .into_iter()
                .map(|(name, server)| (name, server.into()))
                .collect::<BTreeMap<_, _>>(),
        }
    }
}

/// MCP 服务启用状态更新请求。
/// Request payload for updating one MCP server enabled state.
#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct McpServerEnabledRequest {
    pub enabled: bool,
}

fn default_mcp_enabled() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::McpConfigRequest;

    #[test]
    fn accepts_empty_mcp_config_request_like_python_default() {
        let request: McpConfigRequest = serde_json::from_value(serde_json::json!({})).unwrap();

        assert!(request.mcp_servers.is_empty());
    }

    #[test]
    fn rejects_non_object_mcp_env() {
        let result = serde_json::from_value::<McpConfigRequest>(serde_json::json!({
            "mcpServers": {
                "demo": {
                    "transport": "streamable_http",
                    "url": "https://mcp.example.com",
                    "env": ["invalid"]
                }
            }
        }));

        assert!(result.is_err());
    }
}
