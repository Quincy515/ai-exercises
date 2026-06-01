use std::time::Duration;

use anyhow::{anyhow, Result};
use async_openai::{config::OpenAIConfig, Client};
use async_trait::async_trait;
use serde_json::{json, Map, Value};
use tracing::{error, info};

use crate::domain::{
    external::{Llm, LlmMessage, Response, ResponseFormat, Tool, ToolChoice},
    models::LlmConfig,
};

#[derive(Debug, Clone)]
pub struct OpenAILLM {
    client: Client<OpenAIConfig>,
    model_name: String,
    temperature: f32,
    max_tokens: usize,
    timeout: Duration,
}

/// 基于 OpenAI SDK 的 LLM 实现
impl OpenAILLM {
    /// 构造函数，完成 OpenAI 客户端的创建和参数初始化
    pub fn new(llm_config: LlmConfig) -> Self {
        let default_config = LlmConfig::default();
        let base_url = llm_config
            .base_url
            .or(default_config.base_url)
            .unwrap_or_default();
        let api_key = llm_config.api_key.unwrap_or_default();
        let model_name = llm_config
            .model_name
            .or(default_config.model_name)
            .unwrap_or_default();
        let temperature = llm_config
            .temperature
            .or(default_config.temperature)
            .unwrap_or_default();
        let max_tokens = llm_config
            .max_tokens
            .or(default_config.max_tokens)
            .unwrap_or_default();

        Self {
            client: Client::with_config(
                OpenAIConfig::new()
                    .with_api_base(base_url)
                    .with_api_key(api_key),
            ),
            model_name,
            temperature,
            max_tokens,
            timeout: Duration::from_secs(3600),
        }
    }

    fn build_request(
        &self,
        messages: Vec<LlmMessage>,
        tools: Option<Vec<Tool>>,
        response_format: Option<ResponseFormat>,
        tool_choice: Option<ToolChoice>,
    ) -> Value {
        let mut request = Map::from_iter([
            ("model".to_string(), json!(self.model_name)),
            ("temperature".to_string(), json!(self.temperature)),
            ("max_tokens".to_string(), json!(self.max_tokens)),
            (
                "messages".to_string(),
                Value::Array(messages.into_iter().map(Value::Object).collect()),
            ),
        ]);

        if let Some(response_format) = response_format {
            request.insert(
                "response_format".to_string(),
                Value::Object(response_format),
            );
        }

        if let Some(tools) = tools.filter(|tools| !tools.is_empty()) {
            request.insert(
                "tools".to_string(),
                Value::Array(tools.into_iter().map(Value::Object).collect()),
            );
            // 关闭并行工具调用(deepseek 没有该参数)
            request.insert("parallel_tool_calls".to_string(), Value::Bool(false));

            if let Some(tool_choice) = tool_choice {
                request.insert("tool_choice".to_string(), Value::String(tool_choice));
            }
        }

        Value::Object(request)
    }
}

#[async_trait]
impl Llm for OpenAILLM {
    async fn invoke(
        &self,
        messages: Vec<LlmMessage>,
        tools: Option<Vec<Tool>>,
        response_format: Option<ResponseFormat>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<Response> {
        let has_tools = tools.as_ref().is_some_and(|tools| !tools.is_empty());
        let request = self.build_request(messages, tools, response_format, tool_choice);

        if has_tools {
            info!(model = %self.model_name, "调用 OpenAI 客户端向 LLM 发起请求并携带工具信息");
        } else {
            info!(model = %self.model_name, "调用 OpenAI 客户端向 LLM 发起请求未携带工具信息");
        }

        let response: Value =
            match tokio::time::timeout(self.timeout, self.client.chat().create_byot(request)).await
            {
                Ok(Ok(response)) => response,
                Ok(Err(err)) => {
                    error!(error = %err, "调用 OpenAI 客户端发生错误");
                    return Err(anyhow!("调用 OpenAI 客户端向 LLM 发起请求出错"));
                }
                Err(err) => {
                    error!(error = %err, "调用 OpenAI 客户端发生超时");
                    return Err(anyhow!("调用 OpenAI 客户端向 LLM 发起请求超时"));
                }
            };

        info!(response = %response, "OpenAI 客户端返回内容");

        response
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| anyhow!("OpenAI 客户端返回内容缺少 choices[0].message"))
    }

    fn model_name(&self) -> String {
        self.model_name.clone()
    }

    fn temperature(&self) -> f32 {
        self.temperature
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::models::LlmConfig;

    #[test]
    fn builds_request_without_tools() {
        let llm = OpenAILLM::new(LlmConfig {
            base_url: Some("https://api.deepseek.com".to_string()),
            api_key: None,
            model_name: Some("deepseek-chat".to_string()),
            temperature: Some(0.7),
            max_tokens: Some(8192),
        });

        let request = llm.build_request(
            vec![LlmMessage::from_iter([
                ("role".to_string(), "user".into()),
                ("content".to_string(), "Hello".into()),
            ])],
            None,
            None,
            None,
        );

        assert_eq!(request["model"], json!("deepseek-chat"));
        let temperature = request["temperature"].as_f64().unwrap();
        assert!((temperature - 0.7).abs() < 0.000_001);
        assert_eq!(request["max_tokens"], json!(8192));
        assert!(request.get("tools").is_none());
        assert!(request.get("tool_choice").is_none());
        assert!(request.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn builds_request_with_tools() {
        let llm = OpenAILLM::new(LlmConfig::default());
        let tool = Tool::from_iter([
            ("type".to_string(), json!("function")),
            (
                "function".to_string(),
                json!({
                    "name": "search",
                    "description": "Search documents",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }),
            ),
        ]);

        let request = llm.build_request(
            vec![LlmMessage::from_iter([
                ("role".to_string(), "user".into()),
                ("content".to_string(), "Search".into()),
            ])],
            Some(vec![tool]),
            Some(ResponseFormat::from_iter([(
                "type".to_string(),
                json!("json_object"),
            )])),
            Some("auto".to_string()),
        );

        assert_eq!(request["tools"].as_array().map(Vec::len), Some(1));
        assert_eq!(request["tool_choice"], json!("auto"));
        assert_eq!(request["parallel_tool_calls"], json!(false));
        assert_eq!(request["response_format"]["type"], json!("json_object"));
    }

    #[tokio::test]
    #[ignore = "requires LLM_API_KEY and live provider access"]
    async fn invokes_live_llm_when_configured() -> Result<()> {
        let llm = OpenAILLM::new(LlmConfig {
            base_url: std::env::var("LLM_BASE_URL").ok(),
            api_key: std::env::var("LLM_API_KEY").ok(),
            model_name: std::env::var("LLM_MODEL_NAME").ok(),
            temperature: Some(0.7),
            max_tokens: Some(8192),
        });

        let response = llm
            .invoke(
                vec![LlmMessage::from_iter([
                    ("role".to_string(), "user".into()),
                    ("content".to_string(), "Hello, how are you?".into()),
                ])],
                None,
                None,
                None,
            )
            .await?;

        assert!(!response.is_empty());
        Ok(())
    }
}
