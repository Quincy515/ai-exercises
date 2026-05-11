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
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    io::{self, Write},
};
use tokio::io::{AsyncBufReadExt, BufReader};
use validator::Validate;

const MODEL: &str = "gpt-5.4-mini";
const MAX_TOOL_ROUNDS: usize = 8;
type ToolFn = fn(&str) -> String;

const SYSTEM_PROMPT: &str = r#"你是一个智能企业报销助手。你的任务是根据用户的请求和公司的报销政策，帮助员工填写并提交差旅报销单。

公司报销政策：
- 必须先查询员工信息，拿到员工姓名和职级。
- 必须用 calculator 计算交通费、住宿费、餐饮费之和，得到 total_cost。
- total_cost 小于等于 1000 元时，报销级别是“标准”。
- total_cost 大于 1000 元且小于等于 2000 元时，报销级别是“高级”。
- total_cost 大于 2000 元时，报销级别是“VIP”。
- total_cost 大于 2000 元时，只有员工职级为“总监”才可以填“VIP”；其他职级必须填“高级”。

你必须遵循以下思考和行动的循环模式（ReAct）：

1. **思考(Thought)**
   - **回顾目标**：当前最终目标是什么？例如：填写并提交一份完整的报销单。
   - **分析现状**：已经获取了哪些信息？还缺少哪些信息？
   - **运用CoT(Chain-of-Thought)**：仔细阅读并逐步应用报销政策，例如计算总金额、判断报销级别。
   - **规划下一步**：接下来应该查询信息、计算金额、向用户提问，还是准备提交？

2. **行动(Action)**
   - 根据你的思考，决定调用工具或向用户提问。
   - 可用工具有：`get_employee_info`、`submit_reimbursement`、`calculator`。
   - 所有信息都已集齐并计算完毕后，最后一步行动必须调用 `submit_reimbursement` 工具。
   - 调用工具之前请先输出简短思考，给用户持续反馈。

请开始工作。
"#;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut agent = ReActAgent::new().await?;
    agent.chat_loop().await
}

struct ReActAgent {
    client: Client<OpenAIConfig>,
    messages: Vec<ChatCompletionRequestMessage>,
    tools: Vec<ChatCompletionTools>,
    available_tools: HashMap<&'static str, ToolFn>,
}

impl ReActAgent {
    async fn new() -> anyhow::Result<Self> {
        dotenvy::dotenv().ok();

        let tools = vec![
            ChatCompletionTools::Function(GetEmployeeInfoInput::tool()?),
            ChatCompletionTools::Function(SubmitReimbursementInput::tool()?),
            ChatCompletionTools::Function(CalculatorInput::tool()?),
        ];

        Ok(Self {
            client: Client::new(),
            messages: vec![ChatCompletionRequestSystemMessage::from(SYSTEM_PROMPT).into()],
            tools,
            available_tools: HashMap::from([
                (
                    GetEmployeeInfoInput::tool_name(),
                    call_get_employee_info as ToolFn,
                ),
                (
                    SubmitReimbursementInput::tool_name(),
                    call_submit_reimbursement as ToolFn,
                ),
                (CalculatorInput::tool_name(), call_calculator as ToolFn),
            ]),
        })
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
            self.execute_tool_calls(tool_calls)?;
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

    fn execute_tool_calls(
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

            let result = match self.available_tools.get(tool_name.as_str()) {
                Some(function_to_call) => function_to_call(&tool_arguments),
                None => json!({"error": format!("未知工具: {tool_name}")}).to_string(),
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

// 中文：ToolInput 把 Pydantic BaseModel 的公共能力集中到一个 trait。
// English: ToolInput centralizes the shared Pydantic BaseModel-like behavior.
trait ToolInput: Sized + JsonSchema + DeserializeOwned + Validate {
    fn tool_name() -> &'static str;
    fn description() -> &'static str;

    /// 中文：schemars 根据 struct 自动生成 schema，类似 Pydantic 的 model_json_schema()。
    /// English: schemars generates JSON Schema from the struct, similar to model_json_schema().
    fn model_json_schema() -> anyhow::Result<Value> {
        let mut schema = serde_json::to_value(schema_for!(Self))?;
        schema
            .as_object_mut()
            .context("schema 必须是 JSON object")?
            .remove("$schema");
        Ok(schema)
    }

    /// 中文：把输入 struct 转成 OpenAI tool 定义。
    /// English: Convert the input struct into an OpenAI tool definition.
    fn tool() -> anyhow::Result<ChatCompletionTool> {
        Ok(ChatCompletionTool {
            function: FunctionObjectArgs::default()
                .name(Self::tool_name())
                .description(Self::description())
                .parameters(Self::model_json_schema()?)
                .strict(true)
                .build()?,
        })
    }

    /// 中文：解析模型返回的 arguments JSON，并执行 validator 校验。
    /// English: Parse the model-returned arguments JSON and run validator checks.
    fn model_validate_json(arguments: &str) -> anyhow::Result<Self> {
        let input: Self = serde_json::from_str(arguments)
            .with_context(|| format!("工具参数不是有效 JSON: {arguments}"))?;
        input.validate()?;
        Ok(input)
    }
}

macro_rules! impl_tool_input {
    ($ty:ty, $name:literal, $description:literal) => {
        impl ToolInput for $ty {
            fn tool_name() -> &'static str {
                $name
            }

            fn description() -> &'static str {
                $description
            }
        }
    };
}

#[derive(Debug, Clone, Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
struct GetEmployeeInfoInput {
    /// 员工工号。/ Employee ID to query.
    #[validate(length(min = 1))]
    employee_id: String,
}

impl_tool_input!(
    GetEmployeeInfoInput,
    "get_employee_info",
    "根据员工工号查询员工信息，包括姓名和职级"
);

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
struct SubmitReimbursementInput {
    /// 员工工号。/ Employee ID.
    #[validate(length(min = 1))]
    employee_id: String,
    /// 员工姓名。/ Employee name.
    #[validate(length(min = 1))]
    employee_name: String,
    /// 提交日期，格式 YYYY-MM-DD。/ Submission date in YYYY-MM-DD format.
    #[validate(length(min = 1))]
    submission_date: String,
    /// 出差开始日期，格式 YYYY-MM-DD。/ Trip start date in YYYY-MM-DD format.
    #[validate(length(min = 1))]
    trip_start_date: String,
    /// 出差结束日期，格式 YYYY-MM-DD。/ Trip end date in YYYY-MM-DD format.
    #[validate(length(min = 1))]
    trip_end_date: String,
    /// 出差目的地。/ Business trip destination.
    #[validate(length(min = 1))]
    destination: String,
    /// 交通费用。/ Transportation cost.
    #[validate(range(min = 0.0))]
    transportation_cost: f64,
    /// 住宿费用。/ Accommodation cost.
    #[validate(range(min = 0.0))]
    accommodation_cost: f64,
    /// 餐饮费用。/ Meal cost.
    #[validate(range(min = 0.0))]
    meal_cost: f64,
    /// 总报销金额。/ Total reimbursement cost.
    #[validate(range(min = 0.0))]
    total_cost: f64,
    /// 报销级别。/ Reimbursement level.
    reimbursement_level: ReimbursementLevel,
}

impl_tool_input!(
    SubmitReimbursementInput,
    "submit_reimbursement",
    "提交已填写的报销表单"
);

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum ReimbursementLevel {
    #[serde(rename = "标准")]
    Standard,
    #[serde(rename = "高级")]
    Premium,
    #[serde(rename = "VIP")]
    Vip,
}

#[derive(Debug, Clone, Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
struct CalculatorInput {
    /// 数学表达式，例如：'123+456+789'。/ Math expression, for example '123+456+789'.
    #[validate(length(min = 1))]
    expression: String,
}

impl_tool_input!(
    CalculatorInput,
    "calculator",
    "一个简单的计算器，可以执行数学表达式"
);

fn run_typed_tool<T, F>(arguments: &str, function: F) -> String
where
    T: ToolInput,
    F: FnOnce(T) -> String,
{
    match T::model_validate_json(arguments) {
        Ok(input) => function(input),
        Err(error) => json!({"error": format!("工具参数校验失败: {error:#}")}).to_string(),
    }
}

fn call_get_employee_info(arguments: &str) -> String {
    run_typed_tool::<GetEmployeeInfoInput, _>(arguments, |input| {
        get_employee_info(&input.employee_id)
    })
}

fn call_submit_reimbursement(arguments: &str) -> String {
    run_typed_tool::<SubmitReimbursementInput, _>(arguments, submit_reimbursement)
}

fn call_calculator(arguments: &str) -> String {
    run_typed_tool::<CalculatorInput, _>(arguments, |input| calculator(&input.expression))
}

/// 根据员工工号查询员工信息，包括姓名和职级。
/// Query employee information by employee ID, including name and level.
fn get_employee_info(employee_id: &str) -> String {
    println!("--- 正在查询工号 {employee_id} 的信息... ---");
    if employee_id == "E12345" {
        return json!({"name": "张三", "level": "经理"}).to_string();
    }
    json!({"error": "该员工不存在"}).to_string()
}

/// 提交已填写的报销表单。
/// Submit the completed reimbursement form.
fn submit_reimbursement(input: SubmitReimbursementInput) -> String {
    println!("--- 已提交报销信息 ---");

    let form = json!({
        "employee_id": input.employee_id,
        "employee_name": input.employee_name,
        "submission_date": input.submission_date,
        "trip_start_date": input.trip_start_date,
        "trip_end_date": input.trip_end_date,
        "destination": input.destination,
        "transportation_cost": input.transportation_cost,
        "accommodation_cost": input.accommodation_cost,
        "meal_cost": input.meal_cost,
        "total_cost": input.total_cost,
        "reimbursement_level": input.reimbursement_level,
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&form).unwrap_or_else(|_| form.to_string())
    );

    json!({"status": "success", "message": "报销单提交成功"}).to_string()
}

/// 一个简单的计算器，可以执行数学表达式。
/// A simple calculator that evaluates math expressions.
fn calculator(expression: &str) -> String {
    match fasteval::ez_eval(expression, &mut fasteval::EmptyNamespace) {
        Ok(result) if result.is_finite() => json!({"result": result}).to_string(),
        Ok(_) => json!({"error": "无效表达式, 错误信息: 结果不是有限数字"}).to_string(),
        Err(error) => json!({"error": format!("无效表达式, 错误信息: {error}")}).to_string(),
    }
}
