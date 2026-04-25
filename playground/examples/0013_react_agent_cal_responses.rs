use std::{
    collections::HashMap,
    io::{self, Write},
};

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::responses::{
        CreateResponseArgs, EasyInputMessage, FunctionCallOutput, FunctionCallOutputItemParam,
        FunctionTool, FunctionToolCall, InputItem, InputParam, Item, MessageItem, OutputItem,
        Response, Tool,
    },
};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};

const MODEL: &str = "gpt-5.4-mini";
const SYSTEM_PROMPT: &str = "你是一个强大的聊天机器人，请根据用户的提问进行答复，如果需要调用工具请直接调用，不知道请直接回复不清楚";

type ToolFn = fn(&str) -> String;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let mut agent = ReActAgent::new()?;
    agent.chat_loop().await
}

struct ReActAgent {
    client: Client<OpenAIConfig>,
    input_items: Vec<InputItem>,
    tools: Vec<Tool>,
    available_tools: HashMap<&'static str, ToolFn>,
}

impl ReActAgent {
    fn new() -> Result<Self, anyhow::Error> {
        dotenvy::dotenv().ok();

        Ok(Self {
            client: Client::new(),
            input_items: Vec::new(),
            tools: vec![Tool::Function(FunctionTool {
                name: "calculator".to_string(),
                description: Some("一个可以计算数学表达式的计算器".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "需要计算的数学表达式，例如：'123+456+789'"
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": false
                })),
                strict: Some(true),
                defer_loading: None,
            })],
            available_tools: HashMap::from([("calculator", calculator as ToolFn)]),
        })
    }

    async fn process_query(&mut self, query: &str) -> Result<String, anyhow::Error> {
        self.input_items.push(EasyInputMessage::from(query).into());

        let response = self.create_response(true).await?;
        let function_call = find_function_call(&response);

        if let Some(function_call) = function_call {
            let result = self.call_tool(&function_call)?;
            println!("Tool Call: {}", function_call.name);
            println!("Tool [{}] Result: {}", function_call.name, result);

            self.input_items
                .push(Item::FunctionCall(function_call.clone()).into());
            self.input_items.push(
                Item::FunctionCallOutput(FunctionCallOutputItemParam {
                    call_id: function_call.call_id,
                    output: FunctionCallOutput::Text(result),
                    id: None,
                    status: None,
                })
                .into(),
            );

            let second_response = self.create_response(false).await?;
            let content = output_text(&second_response);
            self.remember_response(second_response);
            return Ok(format!("Assistant: {content}"));
        }

        let content = output_text(&response);
        self.remember_response(response);
        Ok(format!("Assistant: {content}"))
    }

    async fn create_response(&self, include_tools: bool) -> Result<Response, anyhow::Error> {
        let mut request = CreateResponseArgs::default();
        request
            .model(MODEL)
            .instructions(SYSTEM_PROMPT)
            .max_output_tokens(512u32)
            .input(InputParam::Items(self.input_items.clone()));

        if include_tools {
            request.tools(self.tools.clone());
        }

        Ok(self.client.responses().create(request.build()?).await?)
    }

    fn call_tool(&self, function_call: &FunctionToolCall) -> Result<String, anyhow::Error> {
        let tool_args: Value = function_call.arguments.parse()?;
        let expression = tool_args["expression"].as_str().unwrap_or_default();
        let function_to_call = self
            .available_tools
            .get(function_call.name.as_str())
            .ok_or_else(|| anyhow::anyhow!("未知工具: {}", function_call.name))?;

        Ok(function_to_call(expression))
    }

    fn remember_response(&mut self, response: Response) {
        for output_item in response.output {
            if let OutputItem::Message(message) = output_item {
                self.input_items
                    .push(Item::Message(MessageItem::Output(message)).into());
            }
        }
    }

    /// 交互式 REPL 循环；输入 `quit` 退出。
    async fn chat_loop(&mut self) -> Result<(), anyhow::Error> {
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
                Ok(resp) => println!("{resp}"),
                Err(e) => eprintln!("\nError: {e:#}"),
            }
        }

        Ok(())
    }
}

fn find_function_call(response: &Response) -> Option<FunctionToolCall> {
    response.output.iter().find_map(|item| match item {
        OutputItem::FunctionCall(function_call) => Some(function_call.clone()),
        _ => None,
    })
}

fn output_text(response: &Response) -> String {
    response.output_text().unwrap_or_default()
}

fn calculator(expression: &str) -> String {
    match fasteval::ez_eval(expression, &mut fasteval::EmptyNamespace) {
        Ok(result) if result.is_finite() => json!({"result": result}).to_string(),
        Ok(_) => json!({"error": "无效表达式, 错误信息: 结果不是有限数字"}).to_string(),
        Err(error) => json!({"error": format!("无效表达式, 错误信息: {error}")}).to_string(),
    }
}
