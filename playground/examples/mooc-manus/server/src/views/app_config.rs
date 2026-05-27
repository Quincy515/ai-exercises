use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::domain::models::LlmConfig;

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
