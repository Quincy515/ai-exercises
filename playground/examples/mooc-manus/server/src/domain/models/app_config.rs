use serde::{Deserialize, Serialize};
use validator::Validate;

/// 语言模型配置
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Validate)]
struct LlmConfig {
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

/// 应用配置信息，包含Agent配置、LLM提供商、A2A网络、MCP服务配置等
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
struct AppConfig {
    /// 语言模型配置
    pub llm_config: LlmConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            llm_config: LlmConfig::default(),
        }
    }
}
