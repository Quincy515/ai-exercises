use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Map, Value};

pub type Message = Map<String, Value>;
pub type Tool = Map<String, Value>;
pub type ResponseFormat = Map<String, Value>;
pub type ToolChoice = String;
pub type Response = Map<String, Value>;

/// 用于 Agent 应用与 LLM 进行交互的接口协议
/// Protocol for Agent applications to interact with LLM providers.
#[async_trait]
pub trait Llm: Send + Sync {
    /// 传递消息列表、工具列表、响应格式、工具选择策略调用 LLM 接口
    /// Invoke the LLM with messages, tools, response format, and tool choice.
    async fn invoke(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<Tool>>,
        response_format: Option<ResponseFormat>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<Response>;

    /// 只读属性，返回 LLM 模型名字
    /// Read-only property, return LLM model name
    fn model_name(&self) -> String;

    /// 只读属性，返回 LLM 温度
    /// Read-only property, return LLM temperature
    fn temperature(&self) -> f32;

    /// 只读属性，返回 LLM 最大输出 tokens 数
    /// Read-only property, return LLM max output tokens
    fn max_tokens(&self) -> usize;
}
