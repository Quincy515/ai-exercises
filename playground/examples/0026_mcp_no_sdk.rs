use anyhow::{Context, anyhow};
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk,
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage, ChatCompletionResponseStream, ChatCompletionTool,
    ChatCompletionTools, CreateChatCompletionRequestArgs, FunctionCall, FunctionCallStream,
    FunctionObjectArgs,
};
use async_openai::{Client, config::OpenAIConfig};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    io::{self, Write},
};
use tokio::io::{AsyncBufReadExt, BufReader};

const MODEL: &str = "gpt-5.4-mini";
const MAX_TOOL_ROUNDS: usize = 8;
const GAODE_MCP_URL_PREFIX: &str = "https://mcp.amap.com/mcp?key=";
const SYSTEM_PROMPT: &str = "你是一个强大的聊天机器人，请根据用户的提问进行回复，如果需要调用工具请直接调用，不知道请直接回复不知道";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut agent = ReActAgent::new().await?;
    agent.chat_loop().await
}

struct ReActAgent {
    client: Client<OpenAIConfig>,
    http_client: reqwest::Client,
    gaode_mcp_url: String,
    messages: Vec<ChatCompletionRequestMessage>,
    tools: Vec<ChatCompletionTools>,
}

impl ReActAgent {
    async fn new() -> anyhow::Result<Self> {
        dotenvy::dotenv().ok();

        let gaode_api_key = gaode_api_key()?;
        let mut agent = Self {
            client: Client::new(),
            http_client: reqwest::Client::new(),
            gaode_mcp_url: format!("{GAODE_MCP_URL_PREFIX}{gaode_api_key}"),
            messages: vec![ChatCompletionRequestSystemMessage::from(SYSTEM_PROMPT).into()],
            tools: Vec::new(),
        };

        agent.init_gaode_mcp().await?;
        Ok(agent)
    }

    /// 中文：调用高德 MCP 的 tools/list，把远程 MCP 工具转换为 OpenAI tool。
    /// English: Call Gaode MCP tools/list and convert remote MCP tools into OpenAI tools.
    async fn init_gaode_mcp(&mut self) -> anyhow::Result<()> {
        let result = self
            .post_gaode_mcp(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }))
            .await?;

        let mcp_tools: Vec<McpTool> = serde_json::from_value(
            result
                .get("tools")
                .cloned()
                .context("高德 MCP tools/list 响应缺少 tools 字段")?,
        )
        .context("解析高德 MCP 工具列表失败")?;

        self.tools = mcp_tools
            .into_iter()
            .map(mcp_tool_to_chat_tool)
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(())
    }

    async fn process_query(&mut self, query: &str) -> anyhow::Result<()> {
        // 中文：只有用户真的输入内容时才追加 user message；工具调用后的下一轮复用当前历史。
        // English: Append a user message only for real input; tool follow-up rounds reuse history.
        if !query.is_empty() {
            self.messages
                .push(ChatCompletionRequestUserMessage::from(query).into());
        }

        for _ in 0..MAX_TOOL_ROUNDS {
            print!("Assistant: ");
            io::stdout().flush()?;

            let (content, tool_calls) = self.stream_assistant_turn().await?;

            if tool_calls.is_empty() {
                self.messages
                    .push(ChatCompletionRequestAssistantMessage::from(content).into());
                println!();
                return Ok(());
            }

            self.push_assistant_tool_message(content, tool_calls.clone())?;
            self.execute_tool_calls(tool_calls).await?;
            println!();
        }

        anyhow::bail!("工具调用轮次超过上限: {MAX_TOOL_ROUNDS}");
    }

    async fn stream_assistant_turn(
        &self,
    ) -> anyhow::Result<(String, Vec<ChatCompletionMessageToolCalls>)> {
        let mut response = self.create_chat_completion_stream().await?;
        let mut content = String::new();
        let mut tool_call_chunks: HashMap<u32, ChatCompletionMessageToolCallChunk> = HashMap::new();

        while let Some(chunk_result) = response.next().await {
            let chunk = chunk_result?;
            let Some(choice) = chunk.choices.first() else {
                continue;
            };

            let delta = &choice.delta;

            if let Some(chunk_content) = delta.content.as_ref() {
                content.push_str(chunk_content);
                print!("{chunk_content}");
                io::stdout().flush()?;
            }

            if let Some(chunk_tool_calls) = &delta.tool_calls {
                merge_tool_call_chunks(&mut tool_call_chunks, chunk_tool_calls);
            }
        }

        Ok((content, build_tool_calls(tool_call_chunks)?))
    }

    fn push_assistant_tool_message(
        &mut self,
        content: String,
        tool_calls: Vec<ChatCompletionMessageToolCalls>,
    ) -> anyhow::Result<()> {
        let mut assistant_message = ChatCompletionRequestAssistantMessageArgs::default();
        if !content.is_empty() {
            assistant_message.content(content);
        }
        assistant_message.tool_calls(tool_calls);
        self.messages.push(assistant_message.build()?.into());
        Ok(())
    }

    async fn execute_tool_calls(
        &mut self,
        tool_calls: Vec<ChatCompletionMessageToolCalls>,
    ) -> anyhow::Result<()> {
        for tool_call in tool_calls {
            let ChatCompletionMessageToolCalls::Function(tool_call) = tool_call else {
                continue;
            };

            let tool_name = tool_call.function.name;
            let tool_arguments = tool_call.function.arguments;

            println!("\nTool Call: {tool_name}");
            println!(
                "Tool Parameters: {}",
                format_tool_arguments(&tool_arguments)
            );

            let result = match serde_json::from_str::<Value>(&tool_arguments) {
                Ok(arguments) => match self.call_gaode_mcp(&tool_name, arguments).await {
                    Ok(result) => result,
                    Err(error) => format!("工具执行出错, Error: {error:#}"),
                },
                Err(error) => format!("工具参数解析出错, Error: {error}"),
            };

            println!("Tool [{tool_name}] Result: {result}");

            self.messages.push(
                ChatCompletionRequestToolMessage {
                    content: result.into(),
                    tool_call_id: tool_call.id,
                }
                .into(),
            );
        }

        Ok(())
    }

    async fn call_gaode_mcp(&self, name: &str, arguments: Value) -> anyhow::Result<String> {
        // 中文：把模型生成的 tool call 转发给高德 MCP 的 tools/call。
        // English: Forward the model-generated tool call to Gaode MCP tools/call.
        let result = self
            .post_gaode_mcp(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments
                }
            }))
            .await?;

        let content = result
            .get("content")
            .and_then(Value::as_array)
            .context("高德 MCP tools/call 响应缺少 content 数组")?;

        let texts: Vec<_> = content
            .iter()
            .filter_map(|item| item.get("text").and_then(Value::as_str))
            .collect();

        if texts.is_empty() {
            return Ok(result.to_string());
        }

        Ok(texts.join("\n"))
    }

    async fn post_gaode_mcp(&self, body: Value) -> anyhow::Result<Value> {
        let response = self
            .http_client
            .post(&self.gaode_mcp_url)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("请求高德 MCP 失败")?
            .error_for_status()
            .context("高德 MCP 返回 HTTP 错误")?;

        let data = response
            .json::<Value>()
            .await
            .context("解析高德 MCP JSON 响应失败")?;

        if let Some(error) = data.get("error") {
            anyhow::bail!("高德 MCP 返回 JSON-RPC 错误: {error}");
        }

        data.get("result")
            .cloned()
            .context("高德 MCP 响应缺少 result 字段")
    }

    async fn create_chat_completion_stream(&self) -> anyhow::Result<ChatCompletionResponseStream> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(MODEL)
            .messages(self.messages.clone())
            .tools(self.tools.clone())
            .stream(true)
            .build()?;

        Ok(self.client.chat().create_stream(request).await?)
    }

    /// 中文：交互式 REPL 循环；输入 `quit` 退出。
    /// English: Interactive REPL loop; enter `quit` to exit.
    async fn chat_loop(&mut self) -> anyhow::Result<()> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);

        loop {
            print!("\nQuery: ");
            io::stdout().flush().ok();

            let mut line = Vec::new();
            if reader.read_until(b'\n', &mut line).await? == 0 {
                break;
            }

            let query = String::from_utf8_lossy(&line);
            let query = query.trim();
            if query.is_empty() {
                continue;
            }
            if query.eq_ignore_ascii_case("quit") {
                break;
            }

            match self.process_query(query).await {
                Ok(()) => {}
                Err(error) => eprintln!("\nError: {error:#}"),
            }
        }

        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct McpTool {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

fn mcp_tool_to_chat_tool(tool: McpTool) -> anyhow::Result<ChatCompletionTools> {
    Ok(ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObjectArgs::default()
            .name(tool.name)
            .description(tool.description)
            .parameters(tool.input_schema)
            .build()?,
    }))
}

fn gaode_api_key() -> anyhow::Result<String> {
    let key = std::env::var("GAODE_API_KEY")
        .or_else(|_| std::env::var("GAODE_KEY"))
        .context("请设置 GAODE_API_KEY 或 GAODE_KEY")?;

    if key.trim().is_empty() {
        anyhow::bail!("GAODE_API_KEY 或 GAODE_KEY 为空");
    }

    Ok(key)
}

fn merge_tool_call_chunks(
    tool_call_chunks: &mut HashMap<u32, ChatCompletionMessageToolCallChunk>,
    chunk_tool_calls: &[ChatCompletionMessageToolCallChunk],
) {
    for chunk_tool_call in chunk_tool_calls {
        let entry = tool_call_chunks
            .entry(chunk_tool_call.index)
            .or_insert_with(|| ChatCompletionMessageToolCallChunk {
                index: chunk_tool_call.index,
                id: None,
                r#type: None,
                function: None,
            });

        if entry.id.is_none() {
            entry.id = chunk_tool_call.id.clone();
        }

        if entry.r#type.is_none() {
            entry.r#type = chunk_tool_call.r#type.clone();
        }

        if let Some(function_delta) = &chunk_tool_call.function {
            let function = entry.function.get_or_insert(FunctionCallStream {
                name: None,
                arguments: None,
            });

            if function.name.is_none() {
                function.name = function_delta.name.clone();
            }

            if let Some(arguments_delta) = &function_delta.arguments {
                function
                    .arguments
                    .get_or_insert_with(String::new)
                    .push_str(arguments_delta);
            }
        }
    }
}

fn build_tool_calls(
    tool_call_chunks: HashMap<u32, ChatCompletionMessageToolCallChunk>,
) -> anyhow::Result<Vec<ChatCompletionMessageToolCalls>> {
    let mut tool_call_chunks: Vec<_> = tool_call_chunks.into_values().collect();
    tool_call_chunks.sort_by_key(|tool_call| tool_call.index);

    tool_call_chunks
        .into_iter()
        .map(|tool_call| {
            let function = tool_call
                .function
                .ok_or_else(|| anyhow!("工具调用缺少 function"))?;

            Ok(ChatCompletionMessageToolCalls::Function(
                ChatCompletionMessageToolCall {
                    id: tool_call.id.ok_or_else(|| anyhow!("工具调用缺少 id"))?,
                    function: FunctionCall {
                        name: function
                            .name
                            .ok_or_else(|| anyhow!("工具调用缺少 function.name"))?,
                        arguments: function.arguments.unwrap_or_default(),
                    },
                },
            ))
        })
        .collect()
}

fn format_tool_arguments(arguments: &str) -> String {
    match serde_json::from_str::<Value>(arguments) {
        Ok(value) => serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string()),
        Err(_) => arguments.to_owned(),
    }
}
