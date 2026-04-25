use std::{
    collections::HashMap,
    io::{self, Write},
};

use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage, ChatCompletionResponseMessage, ChatCompletionTool,
    ChatCompletionToolChoiceOption, CreateChatCompletionRequestArgs, FunctionObjectArgs,
    ToolChoiceOptions,
};
use async_openai::{Client, config::OpenAIConfig};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};

const MODEL: &str = "gpt-5.4-mini";
type ToolFn = fn(&str) -> String;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let mut agent = ReActAgent::new().await?;
    agent.chat_loop().await
}

struct ReActAgent {
    pub client: Client<OpenAIConfig>,
    pub messages: Vec<ChatCompletionRequestMessage>,
    pub tool: ChatCompletionTool,
    pub available_tools: HashMap<&'static str, ToolFn>,
}

impl ReActAgent {
    async fn new() -> Result<Self, anyhow::Error> {
        dotenvy::dotenv().ok();

        let client = Client::new();

        let messages = vec![ChatCompletionRequestSystemMessage::from("你是一个强大的聊天机器人，请根据用户的提问进行答复，如果需要调用工具请直接调用，不知道请直接回复不清楚").into()];

        let tool = ChatCompletionTool {
            function: FunctionObjectArgs::default().name("calculator").description("一个可以计算数学表达式的计算器").parameters(json!({
                    "type":"object",
                    "properties": {"expression":{"type":"string","description":"需要计算的数学表达式，例如：'123+456+789'"}},
                    "required":["expression"],
                    "additionalProperties": false
                }))
                .strict(true)
                .build()?,
        };

        Ok(Self {
            client,
            messages,
            tool,
            available_tools: HashMap::from([("calculator", calculator as ToolFn)]),
        })
    }

    async fn process_query(&mut self, query: &str) -> Result<String, anyhow::Error> {
        // 处理用户输入
        self.messages
            .push(ChatCompletionRequestUserMessage::from(query).into());

        let response_message = self.create_chat_completion(None).await?;

        // 判断是否执行工具调用
        if let Some(tool_calls) = response_message
            .tool_calls
            .clone()
            .filter(|calls| !calls.is_empty())
        {
            // 将模型第一次工具调用回复添加到历史消息中
            self.messages.push(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(tool_calls.clone())
                    .build()?
                    .into(),
            );

            for tool_call in tool_calls {
                let ChatCompletionMessageToolCalls::Function(tool_call) = tool_call else {
                    continue;
                };

                let tool_name = tool_call.function.name.as_str();
                let tool_args: Value = tool_call.function.arguments.parse()?;
                let expression = tool_args["expression"].as_str().unwrap_or_default();
                let function_to_call = self
                    .available_tools
                    .get(tool_name)
                    .ok_or_else(|| anyhow::anyhow!("未知工具: {tool_name}"))?;
                let result = function_to_call(expression);
                println!("Tool Call: {tool_name}");
                println!("Tool [{tool_name}] Result: {result}");

                // 将工具结果添加到历史消息中
                self.messages.push(
                    ChatCompletionRequestToolMessage {
                        content: result.into(),
                        tool_call_id: tool_call.id,
                    }
                    .into(),
                );
            }

            // 再次调用模型，让它基于工具调用的结果生成最终回复内容
            let second_response_message = self
                .create_chat_completion(Some(ChatCompletionToolChoiceOption::Mode(
                    ToolChoiceOptions::None,
                )))
                .await?;
            let content = second_response_message.content.clone().unwrap_or_default();
            self.messages
                .push(ChatCompletionRequestAssistantMessage::from(content.clone()).into());
            return Ok(format!("Assistant: {content}"));
        }

        let content = response_message.content.unwrap_or_default();
        self.messages
            .push(ChatCompletionRequestAssistantMessage::from(content.clone()).into());
        Ok(format!("Assistant: {content}"))
    }

    async fn create_chat_completion(
        &self,
        tool_choice: Option<ChatCompletionToolChoiceOption>,
    ) -> Result<ChatCompletionResponseMessage, anyhow::Error> {
        let mut request = CreateChatCompletionRequestArgs::default();
        request
            .model(MODEL)
            .messages(self.messages.clone())
            .tools(self.tool.clone());

        if let Some(tool_choice) = tool_choice {
            request.tool_choice(tool_choice);
        }

        let response = self.client.chat().create(request.build()?).await?;
        response
            .choices
            .first()
            .map(|choice| choice.message.clone())
            .ok_or_else(|| anyhow::anyhow!("模型没有返回可用回复"))
    }

    /// 交互式 REPL 循环；输入 `quit` 退出。
    async fn chat_loop(&mut self) -> Result<(), anyhow::Error> {
        // 异步读取标准输入，避免阻塞 tokio 运行时
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);

        loop {
            print!("\nQuery: ");
            // 立即刷新提示符，保证在等待输入前可见
            io::stdout().flush().ok();

            // 部分 IDE 控制台可能传入非 UTF-8 字节，按字节读取可以避免 REPL 直接退出。
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

            // 单轮出错不应中断整个会话，只打印错误信息继续下一轮
            match self.process_query(query).await {
                Ok(resp) => println!("{resp}"),
                Err(e) => eprintln!("\nError: {e:#}"),
            }
        }
        Ok(())
    }
}

fn calculator(expression: &str) -> String {
    match fasteval::ez_eval(expression, &mut fasteval::EmptyNamespace) {
        Ok(result) if result.is_finite() => json!({"result": result}).to_string(),
        Ok(_) => json!({"error": "无效表达式, 错误信息: 结果不是有限数字"}).to_string(),
        Err(error) => json!({"error": format!("无效表达式, 错误信息: {error}")}).to_string(),
    }
}
