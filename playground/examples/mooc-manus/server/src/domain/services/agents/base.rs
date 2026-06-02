use std::time::Duration;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::sleep;
use tracing::{error, warn};
use uuid::Uuid;

use crate::domain::{
    external::{JsonParser, Llm, LlmMessage, ResponseFormat},
    models::{
        AgentConfig, ErrorEvent, Event, Memory, Message, MessageEvent, ToolEvent, ToolEventStatus,
        ToolResult,
    },
    services::tools::{BaseTool, ToolArguments, ToolSchema},
};

/// 具体 Agent 可以覆盖的默认属性，对应 Python 基类中的类属性。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentOptions {
    /// 智能体名字
    pub name: String,
    /// 每个智能体都有自己系统预设的 prompt
    pub system_prompt: String,
    /// Agent 的响应格式
    pub format: Option<String>,
    /// 重试间隔，比如 1s 秒重试
    pub retry_interval: Duration,
    /// 强制选择工具
    pub tool_choice: Option<String>,
}

impl AgentOptions {
    /// 使用 Agent 的名字和系统提示词创建基础选项。
    pub fn new(name: impl Into<String>, system_prompt: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            system_prompt: system_prompt.into(),
            ..Self::default()
        }
    }
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            name: String::new(),
            system_prompt: String::new(),
            format: None,
            retry_interval: Duration::from_secs(1),
            tool_choice: None,
        }
    }
}

/// 基础 Agent 智能体，保存每个 Agent 实例的选项、状态和运行依赖。
pub struct BaseAgent {
    options: AgentOptions,
    /// Agent 通用配置
    agent_config: AgentConfig,
    /// 语言模型协议
    llm: Box<dyn Llm>,
    /// Agent 记忆
    memory: Memory,
    /// JSON 输出解析器
    json_parser: Box<dyn JsonParser>,
    /// 工具集
    tools: Vec<Box<dyn BaseTool>>,
}

impl BaseAgent {
    /// 构造函数，完成 Agent 的初始化。
    pub fn new(
        options: AgentOptions,
        agent_config: AgentConfig,
        llm: Box<dyn Llm>,
        memory: Memory,
        json_parser: Box<dyn JsonParser>,
        tools: Vec<Box<dyn BaseTool>>,
    ) -> Self {
        Self {
            options,
            agent_config,
            llm,
            memory,
            json_parser,
            tools,
        }
    }

    /// 返回具体 Agent 的默认属性。
    pub fn options(&self) -> &AgentOptions {
        &self.options
    }

    /// 返回 Agent 通用配置。
    pub fn agent_config(&self) -> &AgentConfig {
        &self.agent_config
    }

    /// 返回记忆。
    pub fn memory(&self) -> &Memory {
        &self.memory
    }

    /// 返回 JSON 输出解析器。
    pub fn json_parser(&self) -> &dyn JsonParser {
        self.json_parser.as_ref()
    }

    /// 压缩 Agent 的记忆。
    pub fn compact_memory(&mut self) {
        self.memory.compact();
    }

    /// Agent 的状态回滚，该函数用于确保 Agent 的消息列表状态是正确的，用于发送新消息、暂停/停止任务、通知用户
    pub fn roll_back(&mut self, message: Message) -> Result<()> {
        // 1. 取出记忆中的最后一条消息，检查是否是工具调用
        let Some(tool_call) = self
            .memory
            .get_last_message()
            .and_then(get_tool_calls)
            .and_then(|tool_calls| tool_calls.first())
        else {
            return Ok(());
        };

        // 2. 取出消息中的工具调用参数，并提取工具名字
        let function_name = tool_call
            .get("function")
            .and_then(Value::as_object)
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str);

        // 4. 判断当前的工具是不是通知用户（message_ask_user)
        if function_name == Some("message_ask_user") {
            self.memory.add_message(LlmMessage::from_iter([
                ("role".to_string(), Value::String("tool".to_string())),
                (
                    "tool_call_id".to_string(),
                    tool_call.get("id").cloned().unwrap_or(Value::Null),
                ),
                (
                    "function_name".to_string(),
                    Value::String("message_ask_user".to_string()),
                ),
                (
                    "content".to_string(),
                    Value::String(serde_json::to_string(&message)?),
                ),
            ]));
        } else {
            // 5. 否则直接删除最后一条消息
            self.memory.roll_back();
        }

        Ok(())
    }

    /// 传递消息和响应格式调用 Agent，返回本轮依次产生的事件。
    pub async fn invoke(&mut self, query: &str, format: Option<&str>) -> Result<Vec<Event>> {
        // 1. 需要判断是否传递了 format
        let format = format
            .map(str::to_owned)
            .or_else(|| self.options.format.clone());

        // 2. 调用语言模型获取响应
        let mut events = Vec::new();
        let mut message = self
            .invoke_llm(vec![text_message("user", query)], format.as_deref())
            .await?;

        // 3. 循环遍历直到最大迭代次数
        let mut reached_max_iterations = true;
        for _ in 0..self.agent_config.max_iterations {
            // 4. 如果响应内容无法调用则表示 LLM 生成了文本回答，这个时候就是最终答案
            let Some(tool_calls) = get_tool_calls(&message) else {
                reached_max_iterations = false;
                break;
            };

            // 5. 循环遍历工具参数并执行
            let mut tool_messages = Vec::new();
            for tool_call in tool_calls {
                // 解析工具调用参数，如果没有 function 参数，直接跳过
                let Some(function) = tool_call.get("function").and_then(Value::as_object) else {
                    continue;
                };
                // 6. 取出调用工具 id、名字、参数信息
                let tool_call_id = tool_call
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(str::to_owned)
                    .unwrap_or_else(|| Uuid::new_v4().to_string());
                let function_name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("工具调用缺少 function.name"))?
                    .to_owned();
                let arguments = function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("工具[{function_name}]调用缺少 function.arguments"))?;
                let function_args = self
                    .json_parser
                    .invoke(arguments, None)
                    .await?
                    .as_object()
                    .cloned()
                    .ok_or_else(|| anyhow!("工具[{function_name}]参数必须是 JSON 对象"))?;

                // 7. 取出 Agent 中对应的工具
                let tool = self.get_tool(&function_name)?;
                let tool_name = tool.name().to_owned();

                // 8. 返回工具即将调用事件
                // 其中 tool_content 比较特殊，需要在具体业务中进行实现
                events.push(Event::Tool(ToolEvent {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    function_name: function_name.clone(),
                    function_args: function_args.clone(),
                    status: ToolEventStatus::Calling,
                    ..ToolEvent::default()
                }));

                // 9. 调用工具并获取结果
                let result = self
                    .invoke_tool(tool, &function_name, function_args.clone())
                    .await;

                // 10. 返回工具调用结果，其中 tool_content 比较特殊，需要在具体业务中进行实现
                events.push(Event::Tool(ToolEvent {
                    tool_call_id: tool_call_id.clone(),
                    tool_name,
                    function_name: function_name.clone(),
                    function_args,
                    function_result: Some(result.clone()),
                    status: ToolEventStatus::Called,
                    ..ToolEvent::default()
                }));

                // 11. 组装工具响应
                tool_messages.push(tool_message(
                    tool_call_id,
                    function_name,
                    serde_json::to_value(result)?,
                ));
            }

            // 12. 所有工具都执行完成后，调用 LLM 获取汇总消息二次提供
            message = self.invoke_llm(tool_messages, None).await?;
        }

        // 13. 超过最大迭代次数后，返回错误事件
        if reached_max_iterations {
            events.push(Event::Error(ErrorEvent {
                error: format!(
                    "Agent迭代超过最大迭代次数: {}, 任务处理失败",
                    self.agent_config.max_iterations
                ),
                ..ErrorEvent::default()
            }));
        }

        // 14. 在指定步骤内完成了迭代则返回消息事件
        if let Some(content) = message.get("content").and_then(Value::as_str) {
            events.push(Event::Message(MessageEvent {
                message: content.to_owned(),
                ..MessageEvent::default()
            }));
        } else {
            events.push(Event::Error(ErrorEvent {
                error: "Agent未能生成有效回复内容".to_string(),
                ..ErrorEvent::default()
            }));
        }

        Ok(events)
    }

    /// 获取 Agent 所有可用工具的参数声明。
    fn get_available_tools(&self) -> Vec<ToolSchema> {
        self.tools
            .iter()
            .flat_map(|tool| tool.get_tools())
            .collect()
    }

    /// 获取对应工具所在的工具集。
    fn get_tool(&self, tool_name: &str) -> Result<&dyn BaseTool> {
        // 循环遍历所有工具包
        self.tools
            .iter()
            .find(|tool| tool.has_tool(tool_name)) // 判断工具包中是否存在该工具
            .map(AsRef::as_ref) // 转换为 trait 对象
            .ok_or_else(|| anyhow!("未知工具: {tool_name}"))
    }

    /// 调用语言模型并处理记忆内容。
    async fn invoke_llm(
        &mut self,
        messages: Vec<LlmMessage>,
        format: Option<&str>,
    ) -> Result<LlmMessage> {
        // 1. 将消息添加到记忆中
        self.add_to_memory(messages);

        // 2. 组装语言模型的响应格式
        let response_format = format.map(|format| {
            ResponseFormat::from_iter([("type".to_string(), Value::String(format.to_string()))])
        });
        let available_tools = self.get_available_tools();
        let tools = (!available_tools.is_empty()).then_some(available_tools);
        let mut last_error = "LLM连续返回空内容".to_string();

        // 3. 循环向 LLM 发起提问直到最大重试次数
        for _ in 0..self.agent_config.max_retries {
            // 4. 调用语言模型获取响应内容
            match self
                .llm
                .invoke(
                    self.memory.get_messages().to_vec(),
                    tools.clone(),
                    response_format.clone(),
                    self.options.tool_choice.clone(),
                )
                .await
            {
                // 5. 处理单次 AI 响应内容，避免空回复
                Ok(message) if is_empty_assistant_message(&message) => {
                    warn!("LLM回复了空内容，执行重试");
                    self.add_to_memory(vec![
                        text_message("assistant", ""),
                        text_message("user", "AI无响应内容，请继续。"),
                    ]);
                    sleep(self.options.retry_interval).await;
                }
                // 6. 取出非空消息并处理工具调用
                Ok(message) => {
                    let filtered_message = filter_llm_message(message);
                    // 9. 将消息添加到记忆中
                    self.add_to_memory(vec![filtered_message.clone()]);
                    return Ok(filtered_message);
                }
                Err(err) => {
                    // 10. 记录日志并睡眠指定的时间
                    last_error = err.to_string();
                    error!(error = %err, "调用语言模型发生错误");
                    sleep(self.options.retry_interval).await;
                }
            }
        }

        Err(anyhow!("调用语言模型失败: {last_error}"))
    }

    /// 传递工具集、工具名字和参数调用指定工具。
    async fn invoke_tool(
        &self,
        tool: &dyn BaseTool,
        tool_name: &str,
        arguments: ToolArguments,
    ) -> ToolResult<Value> {
        let mut last_error = String::new();

        // 1. 执行循环调用工具获取结果
        for _ in 0..self.agent_config.max_retries {
            match tool.invoke(tool_name, arguments.clone()).await {
                Ok(result) => return result,
                Err(err) => {
                    last_error = err.to_string();
                    error!(tool_name, error = %err, "调用工具出错");
                    sleep(self.options.retry_interval).await;
                }
            }
        }

        // 2. 循环最大重试次数后没有结果则将错误作为工具的执行结果，让 LLM 自行处理
        ToolResult {
            success: false,
            message: Some(last_error),
            data: None,
        }
    }

    /// 将对应的信息添加到记忆中。
    fn add_to_memory(&mut self, messages: Vec<LlmMessage>) {
        // 1. 检查记忆的消息列表是否为空，如果为空则需要添加预设 Prompt 作为初始记忆
        if self.memory.empty() {
            self.memory
                .add_message(text_message("system", &self.options.system_prompt));
        }

        // 2. 将正常消息添加到记忆中
        self.memory.add_messages(messages);
    }
}

/// 所有具体 Agent 共同遵守的协议。
///
/// 具体 Agent 通过组合 `BaseAgent` 获取公共状态，并在后续实现自己的工作流程。
#[async_trait]
pub trait Agent: Send + Sync {
    /// 返回基础 Agent。
    fn base(&self) -> &BaseAgent;

    /// 返回可修改的基础 Agent。
    fn base_mut(&mut self) -> &mut BaseAgent;

    /// 返回具体 Agent 的默认属性。
    fn options(&self) -> &AgentOptions {
        self.base().options()
    }

    /// 返回 Agent 通用配置。
    fn agent_config(&self) -> &AgentConfig {
        self.base().agent_config()
    }

    /// 返回记忆。
    fn memory(&self) -> &Memory {
        self.base().memory()
    }

    /// 压缩 Agent 的记忆。
    fn compact_memory(&mut self) {
        self.base_mut().compact_memory();
    }

    /// 回滚 Agent 末尾尚未闭合的工具调用。
    fn roll_back(&mut self, message: Message) -> Result<()> {
        self.base_mut().roll_back(message)
    }

    /// 传递消息和响应格式调用 Agent，返回本轮依次产生的事件。
    async fn invoke(&mut self, query: &str, format: Option<&str>) -> Result<Vec<Event>> {
        self.base_mut().invoke(query, format).await
    }
}

impl Agent for BaseAgent {
    fn base(&self) -> &BaseAgent {
        self
    }

    fn base_mut(&mut self) -> &mut BaseAgent {
        self
    }
}

fn text_message(role: &str, content: &str) -> LlmMessage {
    LlmMessage::from_iter([
        ("role".to_string(), Value::String(role.to_string())),
        ("content".to_string(), Value::String(content.to_string())),
    ])
}

fn tool_message(tool_call_id: String, function_name: String, content: Value) -> LlmMessage {
    LlmMessage::from_iter([
        ("role".to_string(), Value::String("tool".to_string())),
        ("tool_call_id".to_string(), Value::String(tool_call_id)),
        ("function_name".to_string(), Value::String(function_name)),
        ("content".to_string(), content),
    ])
}

fn get_tool_calls(message: &LlmMessage) -> Option<&[Value]> {
    message
        .get("tool_calls")
        .and_then(Value::as_array)
        .filter(|tool_calls| !tool_calls.is_empty())
        .map(Vec::as_slice)
}

fn is_empty_assistant_message(message: &LlmMessage) -> bool {
    message.get("role").and_then(Value::as_str) == Some("assistant")
        && !has_content(message)
        && get_tool_calls(message).is_none()
}

fn has_content(message: &LlmMessage) -> bool {
    message.get("content").is_some_and(|content| match content {
        Value::Null => false,
        Value::String(content) => !content.is_empty(),
        _ => true,
    })
}

fn filter_llm_message(message: LlmMessage) -> LlmMessage {
    // 8. 非 AI 消息则记录日志，并存储 message
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        warn!(
            role = ?message.get("role"),
            "LLM响应内容无法确认消息角色"
        );
        return message;
    }

    // 7. 取出工具调用结果，限制 LLM 一次只调用一个工具
    let mut filtered_message = LlmMessage::from_iter([
        ("role".to_string(), Value::String("assistant".to_string())),
        (
            "content".to_string(),
            message.get("content").cloned().unwrap_or(Value::Null),
        ),
    ]);

    if let Some(tool_calls) = get_tool_calls(&message) {
        filtered_message.insert(
            "tool_calls".to_string(),
            Value::Array(tool_calls.iter().take(1).cloned().collect()),
        );
    }

    filtered_message
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
    };

    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
    use serde_json::{json, Value};

    use super::*;
    use crate::domain::{
        external::{Response, Tool, ToolChoice},
        services::tools::{tool, ToolDefinition},
    };

    type Requests = Arc<Mutex<Vec<Vec<LlmMessage>>>>;
    type ToolCounts = Arc<Mutex<Vec<usize>>>;

    struct MockLlm {
        responses: Mutex<VecDeque<Response>>,
        requests: Requests,
        tool_counts: ToolCounts,
    }

    #[async_trait]
    impl Llm for MockLlm {
        async fn invoke(
            &self,
            messages: Vec<LlmMessage>,
            tools: Option<Vec<Tool>>,
            _response_format: Option<ResponseFormat>,
            _tool_choice: Option<ToolChoice>,
        ) -> Result<Response> {
            self.requests.lock().unwrap().push(messages);
            self.tool_counts
                .lock()
                .unwrap()
                .push(tools.unwrap_or_default().len());
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| anyhow!("缺少 Mock LLM 响应"))
        }

        fn model_name(&self) -> String {
            "mock".to_string()
        }

        fn temperature(&self) -> f32 {
            0.0
        }

        fn max_tokens(&self) -> usize {
            1024
        }
    }

    struct MockJsonParser;

    #[async_trait]
    impl JsonParser for MockJsonParser {
        async fn invoke(&self, text: &str, _default_value: Option<Value>) -> Result<Value> {
            Ok(serde_json::from_str(text)?)
        }
    }

    struct EchoTool {
        definitions: Vec<ToolDefinition>,
        should_fail: bool,
    }

    impl EchoTool {
        fn new(should_fail: bool) -> Self {
            Self {
                definitions: vec![tool(
                    "echo",
                    "回显传入文本",
                    ToolArguments::from_iter([("text".to_string(), json!({"type": "string"}))]),
                    vec!["text".to_string()],
                )],
                should_fail,
            }
        }
    }

    #[async_trait]
    impl BaseTool for EchoTool {
        fn name(&self) -> &str {
            "echo_tool"
        }

        fn tool_definitions(&self) -> &[ToolDefinition] {
            &self.definitions
        }

        async fn call_tool(
            &self,
            tool_name: &str,
            kwargs: ToolArguments,
        ) -> Result<ToolResult<Value>> {
            if self.should_fail {
                return Err(anyhow!("工具执行失败"));
            }

            match tool_name {
                "echo" => Ok(ToolResult {
                    data: Some(Value::Object(kwargs)),
                    ..ToolResult::default()
                }),
                _ => Err(anyhow!("工具[{tool_name}]未找到")),
            }
        }
    }

    fn assistant_message(content: Value) -> LlmMessage {
        LlmMessage::from_iter([
            ("role".to_string(), json!("assistant")),
            ("content".to_string(), content),
        ])
    }

    fn tool_call_message() -> LlmMessage {
        tool_call_message_named("echo")
    }

    fn tool_call_message_named(function_name: &str) -> LlmMessage {
        LlmMessage::from_iter([
            ("role".to_string(), json!("assistant")),
            ("content".to_string(), Value::Null),
            (
                "tool_calls".to_string(),
                json!([{
                    "id": "call-1",
                    "function": {
                        "name": function_name,
                        "arguments": "{\"text\":\"hello\"}"
                    }
                }]),
            ),
        ])
    }

    fn agent(
        responses: Vec<Response>,
        tools: Vec<Box<dyn BaseTool>>,
    ) -> (BaseAgent, Requests, ToolCounts) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let tool_counts = Arc::new(Mutex::new(Vec::new()));
        let llm = MockLlm {
            responses: Mutex::new(VecDeque::from(responses)),
            requests: Arc::clone(&requests),
            tool_counts: Arc::clone(&tool_counts),
        };

        (
            BaseAgent::new(
                AgentOptions {
                    system_prompt: "system prompt".to_string(),
                    retry_interval: Duration::ZERO,
                    ..AgentOptions::default()
                },
                AgentConfig {
                    max_iterations: 3,
                    max_retries: 2,
                    max_search_results: 10,
                },
                Box::new(llm),
                Memory::new(),
                Box::new(MockJsonParser),
                tools,
            ),
            requests,
            tool_counts,
        )
    }

    #[tokio::test]
    async fn invoke_runs_tool_and_records_complete_memory() {
        let (mut agent, requests, tool_counts) = agent(
            vec![
                tool_call_message(),
                assistant_message(json!("final answer")),
            ],
            vec![Box::new(EchoTool::new(false))],
        );

        let events = agent.invoke("echo hello", None).await.unwrap();

        assert_eq!(events.len(), 3);
        let Event::Tool(calling) = &events[0] else {
            panic!("第一个事件必须是工具调用中事件");
        };
        assert_eq!(calling.status, ToolEventStatus::Calling);
        assert_eq!(calling.function_name, "echo");

        let Event::Tool(called) = &events[1] else {
            panic!("第二个事件必须是工具调用完毕事件");
        };
        assert_eq!(called.status, ToolEventStatus::Called);
        assert_eq!(
            called
                .function_result
                .as_ref()
                .and_then(|result| result.data.as_ref())
                .and_then(|data| data.get("text")),
            Some(&json!("hello"))
        );

        let Event::Message(message) = &events[2] else {
            panic!("第三个事件必须是最终消息事件");
        };
        assert_eq!(message.message, "final answer");

        let roles = agent
            .memory()
            .get_messages()
            .iter()
            .filter_map(Memory::get_message_role)
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            vec!["system", "user", "assistant", "tool", "assistant"]
        );
        assert_eq!(requests.lock().unwrap()[0].len(), 2);
        assert_eq!(requests.lock().unwrap()[1].len(), 4);
        assert_eq!(*tool_counts.lock().unwrap(), vec![1, 1]);
    }

    #[tokio::test]
    async fn invoke_retries_empty_assistant_message() {
        let (mut agent, requests, _) = agent(
            vec![
                assistant_message(Value::Null),
                assistant_message(json!("continue")),
            ],
            Vec::new(),
        );

        let events = agent.invoke("hello", None).await.unwrap();

        assert_eq!(events.len(), 1);
        let Event::Message(message) = &events[0] else {
            panic!("事件必须是最终消息事件");
        };
        assert_eq!(message.message, "continue");
        assert_eq!(requests.lock().unwrap().len(), 2);
        let roles = agent
            .memory()
            .get_messages()
            .iter()
            .filter_map(Memory::get_message_role)
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            vec!["system", "user", "assistant", "user", "assistant"]
        );
    }

    #[tokio::test]
    async fn invoke_converts_tool_failure_into_called_event() {
        let (mut agent, _, _) = agent(
            vec![
                tool_call_message(),
                assistant_message(json!("handled failure")),
            ],
            vec![Box::new(EchoTool::new(true))],
        );

        let events = agent.invoke("echo hello", None).await.unwrap();

        let Event::Tool(called) = &events[1] else {
            panic!("第二个事件必须是工具调用完毕事件");
        };
        let result = called.function_result.as_ref().unwrap();
        assert!(!result.success);
        assert_eq!(result.message.as_deref(), Some("工具执行失败"));
    }

    #[tokio::test]
    async fn invoke_reports_iteration_limit_before_returning_last_message() {
        let (mut agent, _, _) = agent(
            vec![
                tool_call_message(),
                tool_call_message(),
                tool_call_message(),
                assistant_message(json!("late answer")),
            ],
            vec![Box::new(EchoTool::new(false))],
        );

        let events = agent.invoke("echo hello", None).await.unwrap();

        assert_eq!(events.len(), 8);
        let Event::Error(error) = &events[6] else {
            panic!("第七个事件必须是最大迭代次数错误");
        };
        assert_eq!(error.error, "Agent迭代超过最大迭代次数: 3, 任务处理失败");
        let Event::Message(message) = &events[7] else {
            panic!("第八个事件必须是最终消息事件");
        };
        assert_eq!(message.message, "late answer");
    }

    #[tokio::test]
    async fn invoke_reports_missing_final_content() {
        let (mut agent, _, _) = agent(
            vec![
                tool_call_message(),
                tool_call_message(),
                tool_call_message(),
                tool_call_message(),
            ],
            vec![Box::new(EchoTool::new(false))],
        );

        let events = agent.invoke("echo hello", None).await.unwrap();

        assert_eq!(events.len(), 8);
        let Event::Error(error) = &events[7] else {
            panic!("第八个事件必须是最终回复内容错误");
        };
        assert_eq!(error.error, "Agent未能生成有效回复内容");
    }

    #[test]
    fn compact_memory_forwards_to_memory() {
        let (mut agent, _, _) = agent(Vec::new(), Vec::new());
        agent.memory.add_message(LlmMessage::from_iter([
            ("role".to_string(), json!("assistant")),
            ("content".to_string(), json!("answer")),
            ("reasoning_content".to_string(), json!("hidden")),
        ]));

        agent.compact_memory();

        assert!(!agent.memory().get_messages()[0].contains_key("reasoning_content"));
    }

    #[test]
    fn roll_back_removes_unfinished_tool_call() {
        let (mut agent, _, _) = agent(Vec::new(), Vec::new());
        agent.memory.add_message(tool_call_message());

        agent.roll_back(Message::default()).unwrap();

        assert!(agent.memory().empty());
    }

    #[test]
    fn roll_back_closes_message_ask_user_tool_call() {
        let (mut agent, _, _) = agent(Vec::new(), Vec::new());
        agent
            .memory
            .add_message(tool_call_message_named("message_ask_user"));

        agent
            .roll_back(Message {
                message: "继续执行".to_string(),
                attachments: vec!["/tmp/report.pdf".to_string()],
            })
            .unwrap();

        let messages = agent.memory().get_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(Memory::get_message_role(&messages[1]), Some("tool"));
        assert_eq!(messages[1].get("tool_call_id"), Some(&json!("call-1")));
        assert_eq!(
            messages[1].get("function_name"),
            Some(&json!("message_ask_user"))
        );
        assert_eq!(
            messages[1].get("content"),
            Some(&json!(
                "{\"message\":\"继续执行\",\"attachments\":[\"/tmp/report.pdf\"]}"
            ))
        );
    }

    #[test]
    fn roll_back_keeps_memory_without_pending_tool_call() {
        let (mut agent, _, _) = agent(Vec::new(), Vec::new());
        agent
            .memory
            .add_message(text_message("assistant", "completed"));

        agent.roll_back(Message::default()).unwrap();

        assert_eq!(agent.memory().get_messages().len(), 1);
    }
}
