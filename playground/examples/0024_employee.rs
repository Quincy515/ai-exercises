use anyhow::anyhow;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk,
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage, ChatCompletionResponseStream, ChatCompletionTool,
    ChatCompletionToolChoiceOption, CreateChatCompletionRequestArgs, FunctionCall,
    FunctionCallStream, FunctionObjectArgs, ToolChoiceOptions,
};
use async_openai::{Client, config::OpenAIConfig};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    io::{self, Write},
};
use tokio::io::{AsyncBufReadExt, BufReader};

const MODEL: &str = "gpt-5.4-mini";
type ToolFn = fn(&str) -> String;
const SYSTEM_PROMPT: &str = r#"你是一个智能企业报销助手。你的任务是根据用户的请求和公司的报销政策，帮助员工填写并提交差旅报销单。

你必须遵循以下思考和行动的循环模式（ReAct）：

1. **思考(Thought)**
   - **回顾目标**：当前我的最终目标是什么？(例如：填写并提交一份完整的报销单)
   - **分析现状**：我已经获取了哪些信息？还缺少哪些信息？
   - **运用CoT(Chain-of-Thought)**：仔细阅读并一步一步地应用报销政策。例如，计算总金额、判断报销级别等。把你的计算和推理过程写在思考中。
   - **规划下一步**：接下来我应该做什么？是查询信息，还是计算，还是准备提交？

2. **行动(Action)**
   - 根据你的思考，决定是调用工具还是向用户提问。
   - 你可用的工具有：`get_employee_info`、`submit_reimbursement`、`calculator`。
   - 如果所有信息都已集齐并计算完毕，你的最后一步行动应该是调用 `submit_reimbursement` 工具。
   - 在调用工具之前也请输出思考过程，不要直接调用，避免造成高延迟体验。

请开始工作。
"#;

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

        let messages = vec![ChatCompletionRequestSystemMessage::from(SYSTEM_PROMPT).into()];

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

    async fn process_query(&mut self, query: &str) -> anyhow::Result<()> {
        // 将用户传递的数据添加到消息列表中
        self.messages
            .push(ChatCompletionRequestUserMessage::from(query).into());
        // Python：print("Assistant: ", end="", flush=True)
        // 读用户输入并显示 Assistant: 提示，不要默认换行，立刻把内容输出到终端
        print!("Assistant: ");
        io::stdout().flush()?;

        // 调用 ai 发起请求
        let mut response = self.create_chat_completion_stream(None).await?;

        // 设置变量，判断是否执行工具调用，组装 content 和 tool_calls
        let mut content = String::new();
        let mut is_tool_calls = false;
        let mut tool_calls_object: HashMap<u32, ChatCompletionMessageToolCallChunk> =
            HashMap::new();

        while let Some(chunk_result) = response.next().await {
            let chunk = chunk_result?;

            let Some(choice) = chunk.choices.first() else {
                continue;
            };

            let delta = &choice.delta;

            // 叠加内容和工具调用
            if let Some(chunk_content) = delta.content.as_ref() {
                content.push_str(chunk_content);
                // 如果是直接生成则流式打印输出的内容
                print!("{chunk_content}");
                io::stdout().flush()?;
            }

            // 判断这轮是不是工具调用
            if let Some(chunk_tool_calls) = &delta.tool_calls {
                is_tool_calls = true;

                for chunk_tool_call in chunk_tool_calls {
                    let entry = tool_calls_object
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
        }

        // 如果是工具调用，则需要将 tool_calls_object 转换成列表
        let mut tool_calls_chunks: Vec<_> = tool_calls_object.into_values().collect();
        tool_calls_chunks.sort_by_key(|tool_call| tool_call.index);

        let tool_calls: Vec<ChatCompletionMessageToolCalls> = tool_calls_chunks
            .into_iter()
            .map(|tool_call| {
                let function = tool_call
                    .function
                    .ok_or_else(|| anyhow::anyhow!("工具调用缺少"))?;

                Ok(ChatCompletionMessageToolCalls::Function(
                    ChatCompletionMessageToolCall {
                        id: tool_call.id.ok_or_else(|| anyhow!("工具调用缺少id"))?,
                        function: FunctionCall {
                            name: function
                                .name
                                .ok_or_else(|| anyhow!("工具调用缺少 function.name"))?,
                            arguments: function.arguments.unwrap_or_default(),
                        },
                    },
                ))
            })
            .collect::<anyhow::Result<_>>()?;

        if tool_calls.is_empty() {
            self.messages
                .push(ChatCompletionRequestAssistantMessage::from(content).into());
            println!();
            return Ok(());
        }

        // 将模型第一次回复的内容添加到历史消息中
        let mut assistant_message = ChatCompletionRequestAssistantMessageArgs::default();
        if !content.is_empty() {
            assistant_message.content(content);
        }
        assistant_message.tool_calls(tool_calls.clone());
        self.messages.push(assistant_message.build()?.into());

        // 循环调用对应的工具
        if is_tool_calls {
            for tool_call in tool_calls {
                let ChatCompletionMessageToolCalls::Function(tool_call) = tool_call else {
                    continue;
                };

                let tool_name = tool_call.function.name.as_str();
                let tool_args: Value = tool_call.function.arguments.parse()?;
                println!("\nTool Call: {tool_name}");
                println!("Tool Parameters: {tool_args}");
                let expression = tool_args["expression"].as_str().unwrap_or_default();
                let function_to_call = self
                    .available_tools
                    .get(tool_name)
                    .ok_or_else(|| anyhow!("未知工具: {tool_name}"))?;

                // 调用工具
                let result = function_to_call(expression);
                println!("Tool [{tool_name}] Result: {result}");

                // 将工具结果添加到历史消息中
                self.messages.push({
                    ChatCompletionRequestToolMessage {
                        content: result.into(),
                        tool_call_id: tool_call.id,
                    }
                    .into()
                });
            }

            // 再次调用模型，让它基于工具调用的结果生成最终回复内容
            let mut second_response = self
                .create_chat_completion_stream(Some(ChatCompletionToolChoiceOption::Mode(
                    ToolChoiceOptions::None,
                )))
                .await?;

            print!("Assistant: ");
            io::stdout().flush()?;

            let mut final_content = String::new();
            while let Some(chunk_result) = second_response.next().await {
                let chunk = chunk_result?;
                let Some(choice) = chunk.choices.first() else {
                    continue;
                };
                if let Some(chunk_content) = choice.delta.content.as_ref() {
                    final_content.push_str(chunk_content);
                    print!("{chunk_content}");
                    io::stdout().flush()?;
                }
            }

            self.messages
                .push(ChatCompletionRequestAssistantMessage::from(final_content).into());
            println!();
        }

        Ok(())
    }
    async fn create_chat_completion_stream(
        &self,
        tool_choice: Option<ChatCompletionToolChoiceOption>,
    ) -> Result<ChatCompletionResponseStream, anyhow::Error> {
        let mut request = CreateChatCompletionRequestArgs::default();
        request
            .model(MODEL)
            .messages(self.messages.clone())
            .tools(self.tool.clone())
            .stream(true);

        if let Some(tool_choice) = tool_choice {
            request.tool_choice(tool_choice);
        }

        let stream = self.client.chat().create_stream(request.build()?).await?;
        Ok(stream)
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
                Ok(()) => {}
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

#[derive(Debug, Clone)]
struct GetEmployeeInfoInput {
    employee_id: String,
}

#[derive(Debug, Clone)]
struct SubmitReimbursementInput {
    employee_id: String,
    employee_name: String,
    submission_date: String,
    trip_start_date: String,
    trip_end_date: String,
    destination: String,
    transportation_cost: f64,
    accommodation_cost: f64,
    meal_cost: f64,
    total_cost: f64,
    reimbursement_level: ReimbursementLevel,
}

#[derive(Debug, Clone)]
enum ReimbursementLevel {
    Standard,
    Manager,
    VIP,
}

#[derive(Debug, Clone)]
struct CalculatorInput {
    expression: String,
}

/// 根据员工工号查询员工信息，包括姓名和职级
fn get_employee_info(employee_id: &str) -> String {
    println!("--- 正在查询工号 {employee_id} 的信息... ---");
    if employee_id == "E12345" {
        return json!({"name": "张三", "level": "经理"}).to_string();
    }
    json!({"error": "未找到该员工"}).to_string()
}

/// 提交已填写的报销表单
fn submit_reimbursement(input: SubmitReimbursementInput) -> String {
    println!("--- 已提交报销信息 ---");
    println!("{input:?}");
    json!({"status": "success", "message": "报销单提交成功"}).to_string()
}
