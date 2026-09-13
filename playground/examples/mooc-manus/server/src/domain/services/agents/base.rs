use std::{sync::Arc, time::Duration};

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
    repositories::SessionRepository,
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
    /// 当前 Agent 所属的会话 id
    session_id: String,
    /// 会话数据仓库，负责读写 Agent 记忆
    session_repository: Arc<dyn SessionRepository>,
    /// Agent 通用配置
    agent_config: AgentConfig,
    /// 语言模型协议
    llm: Arc<dyn Llm>,
    /// Agent 记忆；构造函数不能异步读取，因此第一次使用时再加载
    memory: Option<Memory>,
    /// JSON 输出解析器
    json_parser: Arc<dyn JsonParser>,
    /// 工具集
    tools: Vec<Box<dyn BaseTool>>,
}

impl BaseAgent {
    /// 构造函数，完成 Agent 的初始化。
    pub fn new(
        options: AgentOptions,
        session_id: impl Into<String>,
        session_repository: Arc<dyn SessionRepository>,
        agent_config: AgentConfig,
        llm: Arc<dyn Llm>,
        json_parser: Arc<dyn JsonParser>,
        tools: Vec<Box<dyn BaseTool>>,
    ) -> Self {
        Self {
            options,
            session_id: session_id.into(),
            session_repository,
            agent_config,
            llm,
            memory: None,
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

    /// 返回 JSON 输出解析器。
    pub fn json_parser(&self) -> &dyn JsonParser {
        self.json_parser.as_ref()
    }

    /// 按注册顺序初始化工具，使执行 Agent 直接使用初始化后的工具声明和连接。
    pub(crate) async fn initialize_tools(&mut self) -> Result<()> {
        for tool in &mut self.tools {
            tool.initialize().await?;
        }
        Ok(())
    }

    /// 按指定工具集名称及顺序释放长期资源。
    pub(crate) async fn cleanup_tools(&mut self, names: &[&str]) -> Result<()> {
        for name in names {
            if let Some(tool) = self.tools.iter_mut().find(|tool| tool.name() == *name) {
                tool.cleanup().await?;
            }
        }
        Ok(())
    }

    /// 压缩 Agent 的记忆。
    pub async fn compact_memory(&mut self) -> Result<()> {
        // 1. 先从仓库加载当前 Agent 的记忆
        let mut memory = self.ensure_memory().await?.clone();
        // 2. 压缩后立即持久化，避免重启后恢复出未压缩的数据
        memory.compact();
        self.persist_memory(memory).await
    }

    /// Agent 的状态回滚，该函数用于确保 Agent 的消息列表状态是正确的，用于发送新消息、暂停/停止任务、通知用户
    pub async fn roll_back(&mut self, message: Message) -> Result<()> {
        // 1. 回滚前必须先取得数据库中的完整记忆
        let mut memory = self.ensure_memory().await?.clone();
        // 2. 取出记忆中的最后一条消息，检查是否是工具调用
        let Some(tool_call) = memory
            .get_last_message()
            .and_then(get_tool_calls)
            .and_then(|tool_calls| tool_calls.first())
            .cloned()
        else {
            return Ok(());
        };

        // 3. 取出消息中的工具调用参数，并提取工具名字
        let function_name = tool_call
            .get("function")
            .and_then(Value::as_object)
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str);

        // 4. 判断当前的工具是不是通知用户（message_ask_user)
        if function_name == Some("message_ask_user") {
            memory.add_message(LlmMessage::from_iter([
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
            memory.roll_back();
        }

        // 6. 回滚会改变对话状态，必须同步到数据仓库
        self.persist_memory(memory).await
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

                // message_ask_user 需要等待用户回复，保留待闭合的 assistant 工具调用。
                if function_name == "message_ask_user" {
                    return Ok(events);
                }

                // 11. 组装工具响应
                tool_messages.push(tool_message(
                    tool_call_id,
                    function_name,
                    serde_json::to_string(&result)?,
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
        self.add_to_memory(messages).await?;

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
            // 每次请求都传递完整记忆，使模型能够理解历史上下文。
            let memory_messages = self.ensure_memory().await?.get_messages().to_vec();
            match self
                .llm
                .invoke(
                    memory_messages,
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
                    ])
                    .await?;
                    sleep(self.options.retry_interval).await;
                }
                // 6. 取出非空消息并处理工具调用
                Ok(message) => {
                    let filtered_message = filter_llm_message(message);
                    // 9. 将消息添加到记忆中
                    self.add_to_memory(vec![filtered_message.clone()]).await?;
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

    /// 确保当前 Agent 的记忆已经从会话仓库加载。
    async fn ensure_memory(&mut self) -> Result<&Memory> {
        if self.memory.is_none() {
            let memory = self
                .session_repository
                .get_memory(&self.session_id, &self.options.name)
                .await?;
            self.memory = Some(memory);
        }

        self.memory
            .as_ref()
            .ok_or_else(|| anyhow!("Agent记忆加载失败"))
    }

    /// 保存记忆，并且只在仓库保存成功后更新本地缓存。
    async fn persist_memory(&mut self, memory: Memory) -> Result<()> {
        if let Err(error) = self
            .session_repository
            .save_memory(&self.session_id, &self.options.name, memory.clone())
            .await
        {
            // 保存结果不确定时清空缓存，下次操作重新以仓库数据为准。
            self.memory = None;
            return Err(error);
        }

        self.memory = Some(memory);
        Ok(())
    }

    /// 将对应的信息添加到记忆中。
    async fn add_to_memory(&mut self, messages: Vec<LlmMessage>) -> Result<()> {
        // 1. 先检查并确保记忆存在
        let mut memory = self.ensure_memory().await?.clone();

        // 2. 空记忆先添加系统提示词，只在首次写入时执行
        if memory.empty() {
            memory.add_message(text_message("system", &self.options.system_prompt));
        }

        // 3. 添加本轮消息
        memory.add_messages(messages);

        // 4. 每次修改后都持久化
        self.persist_memory(memory).await
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

    /// 压缩 Agent 的记忆。
    async fn compact_memory(&mut self) -> Result<()> {
        self.base_mut().compact_memory().await
    }

    /// 回滚 Agent 末尾尚未闭合的工具调用。
    async fn roll_back(&mut self, message: Message) -> Result<()> {
        self.base_mut().roll_back(message).await
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

fn tool_message(tool_call_id: String, function_name: String, content: String) -> LlmMessage {
    LlmMessage::from_iter([
        ("role".to_string(), Value::String("tool".to_string())),
        ("tool_call_id".to_string(), Value::String(tool_call_id)),
        ("function_name".to_string(), Value::String(function_name)),
        ("content".to_string(), Value::String(content)),
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
        services::{
            agents::test_support::MemoryRepository,
            tools::{tool, ToolDefinition},
        },
    };

    type Requests = Arc<Mutex<Vec<Vec<LlmMessage>>>>;
    type ToolCounts = Arc<Mutex<Vec<usize>>>;
    const SESSION_ID: &str = "session-1";

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

    struct LifecycleTool {
        name: &'static str,
        definitions: Vec<ToolDefinition>,
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl LifecycleTool {
        fn new(name: &'static str, calls: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                name,
                definitions: Vec::new(),
                calls,
            }
        }
    }

    #[async_trait]
    impl BaseTool for LifecycleTool {
        fn name(&self) -> &str {
            self.name
        }

        fn tool_definitions(&self) -> &[ToolDefinition] {
            &self.definitions
        }

        async fn initialize(&mut self) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("initialize:{}", self.name));
            self.definitions = vec![tool(
                format!("{}_call", self.name),
                "初始化后加载的工具",
                ToolArguments::new(),
                Vec::new(),
            )];
            Ok(())
        }

        async fn call_tool(
            &self,
            _tool_name: &str,
            _kwargs: ToolArguments,
        ) -> Result<ToolResult<Value>> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("call:{}", self.name));
            Ok(ToolResult {
                data: Some(json!(self.name)),
                ..ToolResult::default()
            })
        }

        async fn cleanup(&mut self) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("cleanup:{}", self.name));
            self.definitions.clear();
            Ok(())
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
    ) -> (BaseAgent, Requests, ToolCounts, Arc<MemoryRepository>) {
        let repository = Arc::new(MemoryRepository::default());
        let (agent, requests, tool_counts) =
            agent_with_repository(responses, tools, repository.clone());
        (agent, requests, tool_counts, repository)
    }

    fn agent_with_repository(
        responses: Vec<Response>,
        tools: Vec<Box<dyn BaseTool>>,
        repository: Arc<MemoryRepository>,
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
                    name: "base".to_string(),
                    system_prompt: "system prompt".to_string(),
                    retry_interval: Duration::ZERO,
                    ..AgentOptions::default()
                },
                SESSION_ID,
                repository,
                AgentConfig {
                    max_iterations: 3,
                    max_retries: 2,
                    max_search_results: 10,
                },
                Arc::new(llm),
                Arc::new(MockJsonParser),
                tools,
            ),
            requests,
            tool_counts,
        )
    }

    #[tokio::test]
    async fn initialized_tool_definitions_are_available_to_the_same_agent() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (mut agent, _, _, _) = agent(
            Vec::new(),
            vec![Box::new(LifecycleTool::new("mcp", calls.clone()))],
        );
        assert!(agent.get_available_tools().is_empty());
        assert!(agent.get_tool("mcp_call").is_err());

        agent.initialize_tools().await.unwrap();

        assert_eq!(
            agent.get_available_tools()[0]["function"]["name"],
            "mcp_call"
        );
        let result = agent
            .get_tool("mcp_call")
            .unwrap()
            .invoke("mcp_call", ToolArguments::new())
            .await
            .unwrap();
        assert_eq!(result.data, Some(json!("mcp")));
        assert_eq!(*calls.lock().unwrap(), ["initialize:mcp", "call:mcp"]);
    }

    #[tokio::test]
    async fn tool_cleanup_follows_requested_order_and_scope() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tools = ["a2a", "browser", "mcp"]
            .into_iter()
            .map(|name| Box::new(LifecycleTool::new(name, calls.clone())) as Box<dyn BaseTool>)
            .collect();
        let (mut agent, _, _, _) = agent(Vec::new(), tools);

        agent.cleanup_tools(&["mcp", "a2a"]).await.unwrap();

        assert_eq!(*calls.lock().unwrap(), ["cleanup:mcp", "cleanup:a2a"]);
    }

    #[tokio::test]
    async fn invoke_runs_tool_and_records_complete_memory() {
        let (mut agent, requests, tool_counts, repository) = agent(
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

        let memory = repository.memory(SESSION_ID, "base");
        let roles = memory
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
        let tool_content = requests.lock().unwrap()[1][3]
            .get("content")
            .and_then(Value::as_str)
            .expect("工具消息 content 必须是 JSON 字符串")
            .to_owned();
        assert_eq!(
            serde_json::from_str::<Value>(&tool_content).unwrap()["success"],
            true
        );
        assert_eq!(*tool_counts.lock().unwrap(), vec![1, 1]);
        assert_eq!(
            repository.reads.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[tokio::test]
    async fn invoke_retries_empty_assistant_message() {
        let (mut agent, requests, _, repository) = agent(
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
        let memory = repository.memory(SESSION_ID, "base");
        let roles = memory
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
        let (mut agent, _, _, _) = agent(
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
        let (mut agent, _, _, _) = agent(
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
        let (mut agent, _, _, _) = agent(
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

    #[tokio::test]
    async fn compact_memory_is_persisted() {
        let (mut agent, _, _, repository) = agent(Vec::new(), Vec::new());
        let mut memory = Memory::new();
        memory.add_message(LlmMessage::from_iter([
            ("role".to_string(), json!("assistant")),
            ("content".to_string(), json!("answer")),
            ("reasoning_content".to_string(), json!("hidden")),
        ]));
        repository.insert(SESSION_ID, "base", memory);

        agent.compact_memory().await.unwrap();

        assert!(!repository.memory(SESSION_ID, "base").get_messages()[0]
            .contains_key("reasoning_content"));
    }

    #[tokio::test]
    async fn roll_back_removes_and_persists_unfinished_tool_call() {
        let (mut agent, _, _, repository) = agent(Vec::new(), Vec::new());
        let mut memory = Memory::new();
        memory.add_message(tool_call_message());
        repository.insert(SESSION_ID, "base", memory);

        agent.roll_back(Message::default()).await.unwrap();

        assert!(repository.memory(SESSION_ID, "base").empty());
    }

    #[tokio::test]
    async fn roll_back_closes_and_persists_message_ask_user_tool_call() {
        let (mut agent, _, _, repository) = agent(Vec::new(), Vec::new());
        let mut memory = Memory::new();
        memory.add_message(tool_call_message_named("message_ask_user"));
        repository.insert(SESSION_ID, "base", memory);

        agent
            .roll_back(Message {
                message: "继续执行".to_string(),
                attachments: vec!["/tmp/report.pdf".to_string()],
            })
            .await
            .unwrap();

        let memory = repository.memory(SESSION_ID, "base");
        let messages = memory.get_messages();
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

    #[tokio::test]
    async fn roll_back_keeps_memory_without_pending_tool_call() {
        let (mut agent, _, _, repository) = agent(Vec::new(), Vec::new());
        let mut memory = Memory::new();
        memory.add_message(text_message("assistant", "completed"));
        repository.insert(SESSION_ID, "base", memory);

        agent.roll_back(Message::default()).await.unwrap();

        assert_eq!(
            repository.memory(SESSION_ID, "base").get_messages().len(),
            1
        );
    }

    #[tokio::test]
    async fn new_agent_restores_history_before_calling_llm() {
        let repository = Arc::new(MemoryRepository::default());
        let mut old_memory = Memory::new();
        old_memory.add_messages(vec![
            text_message("system", "system prompt"),
            text_message("user", "old question"),
            text_message("assistant", "old answer"),
        ]);
        repository.insert(SESSION_ID, "base", old_memory);
        let (mut agent, requests, _) = agent_with_repository(
            vec![assistant_message(json!("new answer"))],
            Vec::new(),
            repository.clone(),
        );

        agent.invoke("new question", None).await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests[0].len(), 4);
        assert_eq!(requests[0][1].get("content"), Some(&json!("old question")));
        let memory = repository.memory(SESSION_ID, "base");
        let system_message_count = memory
            .get_messages()
            .iter()
            .filter(|message| Memory::get_message_role(message) == Some("system"))
            .count();
        assert_eq!(system_message_count, 1);
    }

    #[tokio::test]
    async fn repository_errors_stop_the_llm_call_and_allow_reload() {
        let (mut agent, requests, _, repository) =
            agent(vec![assistant_message(json!("answer"))], Vec::new());
        repository
            .fail_save
            .store(true, std::sync::atomic::Ordering::SeqCst);

        let error = agent.invoke("hello", None).await.unwrap_err();

        assert!(error.to_string().contains("模拟保存记忆失败"));
        assert!(requests.lock().unwrap().is_empty());

        // 保存失败会清空缓存，重试时重新读取仓库并可以继续执行。
        let events = agent.invoke("hello again", None).await.unwrap();
        assert!(matches!(events[0], Event::Message(_)));
        assert_eq!(
            repository.reads.load(std::sync::atomic::Ordering::SeqCst),
            2
        );
    }

    #[tokio::test]
    async fn memory_read_error_stops_before_calling_llm() {
        let (mut agent, requests, _, repository) =
            agent(vec![assistant_message(json!("answer"))], Vec::new());
        repository
            .fail_read
            .store(true, std::sync::atomic::Ordering::SeqCst);

        let error = agent.invoke("hello", None).await.unwrap_err();

        assert!(error.to_string().contains("模拟读取记忆失败"));
        assert!(requests.lock().unwrap().is_empty());
        assert_eq!(
            repository.writes.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }
}
