use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::domain::models::{AgentConfig, LlmConfig};

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
