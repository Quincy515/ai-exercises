use serde::{Deserialize, Serialize};
use validator::Validate;

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

/// 应用配置信息，包含Agent配置、LLM提供商、A2A网络、MCP服务配置等
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct AppConfig {
    /// 语言模型配置
    #[serde(default)]
    pub llm_config: LlmConfig,
    /// Agent 通用配置
    #[serde(default)]
    pub agent_config: AgentConfig,
}

#[cfg(test)]
mod tests {
    use super::{AgentConfig, AppConfig, LlmConfig};
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
    fn loads_legacy_config_without_agent_settings() {
        let config: AppConfig = serde_json::from_value(serde_json::json!({
            "llm_config": LlmConfig::default()
        }))
        .unwrap();

        assert_eq!(config.agent_config, AgentConfig::default());
    }
}
