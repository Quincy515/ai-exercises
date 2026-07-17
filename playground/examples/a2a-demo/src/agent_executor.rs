use std::{env, error::Error};

use a2a::event::StreamResponse;
use a2a::{A2AError, Message, Part, Role};
use a2a_server::{AgentExecutor, ExecutorContext};
use async_openai::{Client, config::OpenAIConfig};
use futures::stream::{self, BoxStream};
use serde_json::{Value, json};

const DEEPSEEK_API_BASE: &str = "https://api.deepseek.com";
const DEEPSEEK_MODEL: &str = "deepseek-reasoner";

type AgentResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Clone, Copy)]
struct DeepSeekAgent;

impl DeepSeekAgent {
    async fn invoke(&self, query: &str) -> AgentResult<String> {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(DEEPSEEK_API_BASE)
                .with_api_key(env::var("DEEPSEEK_API_KEY")?),
        );
        let response: Value = client
            .chat()
            .create_byot(json!({
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": query}],
            }))
            .await?;

        Ok(format_answer(&response))
    }
}

pub struct DeepSeekAgentExecutor {
    agent: DeepSeekAgent,
}

impl DeepSeekAgentExecutor {
    pub fn new() -> Self {
        Self {
            agent: DeepSeekAgent,
        }
    }
}

impl AgentExecutor for DeepSeekAgentExecutor {
    fn execute(
        &self,
        context: ExecutorContext,
    ) -> BoxStream<'static, Result<StreamResponse, A2AError>> {
        // 用户传递的数据在 context.message 中，读取第一个文本 Part。
        let query = match extract_query(context.message.as_ref()) {
            Ok(query) => query,
            Err(error) => return Box::pin(stream::once(async move { Err(error) })),
        };
        let agent = self.agent;

        Box::pin(stream::once(async move {
            let answer = agent
                .invoke(&query)
                .await
                .map_err(|error| A2AError::internal(error.to_string()))?;
            Ok(StreamResponse::Message(Message::new(
                Role::Agent,
                vec![Part::text(answer)],
            )))
        }))
    }

    fn cancel(
        &self,
        context: ExecutorContext,
    ) -> BoxStream<'static, Result<StreamResponse, A2AError>> {
        // 暂不支持取消。
        let task_id = context.task_id;
        Box::pin(stream::once(async move {
            Err(A2AError::task_not_cancelable(&task_id))
        }))
    }
}

fn extract_query(message: Option<&Message>) -> Result<String, A2AError> {
    let query = message
        .and_then(|message| message.parts.first())
        .and_then(Part::as_text)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| A2AError::invalid_params("请求消息必须包含一个非空文本 Part"))?;
    Ok(query.to_string())
}

fn format_answer(response: &Value) -> String {
    let message = &response["choices"][0]["message"];
    let reasoning_content = message["reasoning_content"].as_str().unwrap_or_default();
    let content = message["content"].as_str().unwrap_or_default();
    format!("推理内容: {reasoning_content}\n\n答案: {content}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_first_text_part_like_python() {
        let message = Message::new(
            Role::User,
            vec![Part::text("  第一段  "), Part::text("第二段")],
        );

        assert_eq!(extract_query(Some(&message)).unwrap(), "第一段");
    }

    #[test]
    fn rejects_message_without_text() {
        assert!(extract_query(None).is_err());
    }

    #[test]
    fn formats_reasoning_and_answer() {
        let response = json!({
            "choices": [{
                "message": {
                    "reasoning_content": "先分析公式",
                    "content": "结果是 15130"
                }
            }]
        });

        assert_eq!(
            format_answer(&response),
            "推理内容: 先分析公式\n\n答案: 结果是 15130"
        );
    }
}
