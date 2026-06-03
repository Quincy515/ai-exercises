use anyhow::Result;
use serde_json::Value;
use tracing::info;

use crate::domain::{
    external::{JsonParser, Llm},
    models::{
        AgentConfig, Event, ExecutionStatus, File, Memory, Message, MessageEvent, MessageRole,
        Plan, Step, StepEvent, StepEventStatus, ToolEventStatus, WaitEvent,
    },
    services::{
        prompts::{EXECUTION_PROMPT, REACT_SYSTEM_PROMPT, SUMMARIZE_PROMPT, SYSTEM_PROMPT},
        tools::BaseTool,
    },
};

use super::{Agent, AgentOptions, BaseAgent};

/// 基于 ReAct 架构的执行 Agent。
pub struct ReActAgent {
    base: BaseAgent,
}

impl ReActAgent {
    /// 创建 ReAct Agent，并固定执行场景使用的选项。
    pub fn new(
        agent_config: AgentConfig,
        llm: Box<dyn Llm>,
        memory: Memory,
        json_parser: Box<dyn JsonParser>,
        tools: Vec<Box<dyn BaseTool>>,
    ) -> Self {
        Self {
            base: BaseAgent::new(
                react_options(),
                agent_config,
                llm,
                memory,
                json_parser,
                tools,
            ),
        }
    }

    /// 根据传递的消息 + 规划 + 子步骤执行相应的子步骤
    pub async fn execute_step(
        &mut self,
        plan: &Plan,
        step: &mut Step,
        message: &Message,
    ) -> Result<Vec<Event>> {
        // 1. 根据传递的内容生成执行消息
        let query = EXECUTION_PROMPT
            .replace("{message}", &message.message)
            .replace("{attachments}", &message.attachments.join("\n"))
            .replace("{language}", &plan.language)
            .replace("{step}", &step.description);

        // 2. 更新步骤的执行状态为运行中并返回 Step 事件
        step.status = ExecutionStatus::Running;
        let mut output = vec![Event::Step(StepEvent {
            step: step.clone(),
            status: StepEventStatus::Started,
            ..StepEvent::default()
        })];

        // 3. 调用 invoke() 获取 Agent 返回的事件内容
        let events = self.base.invoke(&query, None).await?;
        output.reserve(events.len());

        for event in events {
            // 4. 根据事件类型执行不同操作
            match event {
                // 5. 工具事件需要判断工具的名称是否为 message_ask_user
                Event::Tool(tool_event) if tool_event.function_name == "message_ask_user" => {
                    // 6. 工具如果在调用中，我们需要返回一条消息告知用户需要让用户处理什么
                    if tool_event.status == ToolEventStatus::Calling {
                        // todo: 由于 message_ask_user 工具还未实现，所以参数未定，暂时定为 text
                        let message = tool_event
                            .function_args
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned();
                        output.push(Event::Message(MessageEvent {
                            role: MessageRole::Assistant,
                            message,
                            ..MessageEvent::default()
                        }));
                    } else {
                        // 7. 如果工具事件为已调用，则需要返回等待事件并中断程序
                        output.push(Event::Wait(WaitEvent::default()));
                        return Ok(output);
                    }
                }
                Event::Message(message_event) => {
                    // 8. 返回消息事件，意味着 content 有内容，则代表执行 Agent 已运行完毕
                    step.status = ExecutionStatus::Completed;
                    // 9. message 中输出的数据结构为 json，需要提取并解析
                    let parsed_obj = self
                        .base
                        .json_parser()
                        .invoke(&message_event.message, None)
                        .await?;
                    let new_step: Step = serde_json::from_value(parsed_obj)?;

                    // 10. 使用结构化输出更新当前子步骤的数据
                    step.success = new_step.success;
                    step.result = new_step.result;
                    step.attachments = new_step.attachments;

                    // 11. 返回步骤完成事件
                    output.push(Event::Step(StepEvent {
                        step: step.clone(),
                        status: StepEventStatus::Completed,
                        ..StepEvent::default()
                    }));

                    // 12. 子步骤存在结果时，将结果消息返回给用户
                    if let Some(result) = step.result.as_deref().filter(|result| !result.is_empty())
                    {
                        output.push(Event::Message(MessageEvent {
                            role: MessageRole::Assistant,
                            message: result.to_owned(),
                            ..MessageEvent::default()
                        }));
                    }
                }
                Event::Error(error_event) => {
                    // 13. 错误事件更新步骤状态和错误信息
                    step.status = ExecutionStatus::Failed;
                    step.error = Some(error_event.error.clone());

                    // 14. 返回子步骤对应事件
                    output.push(Event::Step(StepEvent {
                        step: step.clone(),
                        status: StepEventStatus::Failed,
                        ..StepEvent::default()
                    }));
                    output.push(Event::Error(error_event));
                }

                // 15. 其他事件直接返回
                event => output.push(event),
            }
        }

        // 16. 仅在步骤仍处于运行中时，将自然结束的迭代标记为完成
        if step.status == ExecutionStatus::Running {
            step.status = ExecutionStatus::Completed;
        }

        Ok(output)
    }

    /// 调用 Agent 汇总历史消息并生成最终回复 + 附件
    pub async fn summarize(&mut self) -> Result<Vec<Event>> {
        // 1. 使用汇总 Prompt 调用 Agent 生成事件
        let events = self.base.invoke(SUMMARIZE_PROMPT, None).await?;
        let mut output = Vec::with_capacity(events.len());

        for event in events {
            match event {
                // 2. MessageEvent 表示 Agent 已经生成结构化汇总内容
                Event::Message(message_event) => {
                    // 3. 记录日志并解析输出内容
                    info!(message = %message_event.message, "执行Agent生成汇总内容");
                    let parsed_obj = self
                        .base
                        .json_parser()
                        .invoke(&message_event.message, None)
                        .await?;

                    // 4. 将解析数据转换为 Message，并将路径转换为 File 附件
                    let message: Message = serde_json::from_value(parsed_obj)?;
                    let attachments = message
                        .attachments
                        .into_iter()
                        .map(|filepath| File {
                            filepath,
                            ..File::default()
                        })
                        .collect();

                    // 5. 返回最终消息事件
                    output.push(Event::Message(MessageEvent {
                        role: MessageRole::Assistant,
                        message: message.message,
                        attachments,
                        ..MessageEvent::default()
                    }));
                }
                // 6. 其他事件直接返回
                event => output.push(event),
            }
        }

        Ok(output)
    }
}

impl Agent for ReActAgent {
    fn base(&self) -> &BaseAgent {
        &self.base
    }

    fn base_mut(&mut self) -> &mut BaseAgent {
        &mut self.base
    }
}

fn react_options() -> AgentOptions {
    AgentOptions {
        name: "react".to_string(),
        system_prompt: format!("{SYSTEM_PROMPT}\n{REACT_SYSTEM_PROMPT}"),
        // format 控制的是 content，工具调用控制的是 tool_calls，两者可以同时使用。
        format: Some("json_object".to_string()),
        ..AgentOptions::default()
    }
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
        external::{LlmMessage, Response, ResponseFormat, Tool, ToolChoice},
        models::ToolResult,
        services::tools::{tool, ToolArguments, ToolDefinition},
    };

    type Requests = Arc<Mutex<Vec<LlmRequest>>>;

    #[derive(Debug)]
    struct LlmRequest {
        messages: Vec<LlmMessage>,
        tools: Option<Vec<Tool>>,
        response_format: Option<ResponseFormat>,
        tool_choice: Option<ToolChoice>,
    }

    struct MockLlm {
        responses: Mutex<VecDeque<Response>>,
        requests: Requests,
    }

    #[async_trait]
    impl Llm for MockLlm {
        async fn invoke(
            &self,
            messages: Vec<LlmMessage>,
            tools: Option<Vec<Tool>>,
            response_format: Option<ResponseFormat>,
            tool_choice: Option<ToolChoice>,
        ) -> Result<Response> {
            self.requests.lock().unwrap().push(LlmRequest {
                messages,
                tools,
                response_format,
                tool_choice,
            });
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

    struct MessageAskUserTool {
        definitions: Vec<ToolDefinition>,
    }

    impl MessageAskUserTool {
        fn new() -> Self {
            Self {
                definitions: vec![tool(
                    "message_ask_user",
                    "向用户提问并等待回复",
                    ToolArguments::from_iter([("text".to_string(), json!({"type": "string"}))]),
                    vec!["text".to_string()],
                )],
            }
        }
    }

    #[async_trait]
    impl BaseTool for MessageAskUserTool {
        fn name(&self) -> &str {
            "message"
        }

        fn tool_definitions(&self) -> &[ToolDefinition] {
            &self.definitions
        }

        async fn call_tool(
            &self,
            tool_name: &str,
            _kwargs: ToolArguments,
        ) -> Result<ToolResult<Value>> {
            match tool_name {
                "message_ask_user" => Ok(ToolResult::default()),
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

    fn message_ask_user_call(text: &str) -> LlmMessage {
        LlmMessage::from_iter([
            ("role".to_string(), json!("assistant")),
            ("content".to_string(), Value::Null),
            (
                "tool_calls".to_string(),
                json!([{
                    "id": "call-1",
                    "function": {
                        "name": "message_ask_user",
                        "arguments": serde_json::to_string(&json!({"text": text})).unwrap()
                    }
                }]),
            ),
        ])
    }

    fn react(responses: Vec<Response>, tools: Vec<Box<dyn BaseTool>>) -> (ReActAgent, Requests) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let llm = MockLlm {
            responses: Mutex::new(VecDeque::from(responses)),
            requests: Arc::clone(&requests),
        };
        let react = ReActAgent::new(
            AgentConfig {
                max_iterations: 3,
                max_retries: 1,
                max_search_results: 10,
            },
            Box::new(llm),
            Memory::new(),
            Box::new(MockJsonParser),
            tools,
        );
        (react, requests)
    }

    #[test]
    fn react_options_match_python_defaults() {
        let options = react_options();

        assert_eq!(options.name, "react");
        assert_eq!(options.format.as_deref(), Some("json_object"));
        assert!(options.tool_choice.is_none());
        assert!(options.system_prompt.contains(SYSTEM_PROMPT));
        assert!(options.system_prompt.contains(REACT_SYSTEM_PROMPT));
    }

    #[tokio::test]
    async fn execute_step_updates_step_and_returns_result_message() {
        let (mut react, requests) = react(
            vec![assistant_message(json!(
                r#"{
                    "success":true,
                    "result":"数据清洗已完成",
                    "attachments":["/tmp/report.md"]
                }"#
            ))],
            Vec::new(),
        );
        let plan = Plan {
            language: "中文".to_string(),
            ..Plan::default()
        };
        let mut step = Step {
            id: "step-1".to_string(),
            description: "清洗销售数据".to_string(),
            ..Step::default()
        };
        let message = Message {
            message: "请处理销售数据".to_string(),
            attachments: vec!["sales.csv".to_string(), "rules.md".to_string()],
        };

        let events = react
            .execute_step(&plan, &mut step, &message)
            .await
            .unwrap();

        assert_eq!(events.len(), 3);
        let Event::Step(started) = &events[0] else {
            panic!("第一个事件必须是步骤开始事件");
        };
        assert_eq!(started.status, StepEventStatus::Started);
        assert_eq!(started.step.status, ExecutionStatus::Running);

        let Event::Step(completed) = &events[1] else {
            panic!("第二个事件必须是步骤完成事件");
        };
        assert_eq!(completed.status, StepEventStatus::Completed);
        assert_eq!(completed.step.status, ExecutionStatus::Completed);
        assert!(completed.step.success);
        assert_eq!(completed.step.result.as_deref(), Some("数据清洗已完成"));
        assert_eq!(completed.step.attachments, vec!["/tmp/report.md"]);

        let Event::Message(result) = &events[2] else {
            panic!("第三个事件必须是结果消息事件");
        };
        assert_eq!(result.role, MessageRole::Assistant);
        assert_eq!(result.message, "数据清洗已完成");

        assert_eq!(step.status, ExecutionStatus::Completed);
        assert!(step.success);
        assert_eq!(step.result.as_deref(), Some("数据清洗已完成"));
        assert_eq!(step.attachments, vec!["/tmp/report.md"]);

        let requests = requests.lock().unwrap();
        let request = &requests[0];
        assert!(request.tools.is_none());
        assert_eq!(
            request
                .response_format
                .as_ref()
                .and_then(|format| format.get("type")),
            Some(&json!("json_object"))
        );
        assert!(request.tool_choice.is_none());
        let query = request
            .messages
            .last()
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .unwrap();
        assert!(query.contains("请处理销售数据"));
        assert!(query.contains("sales.csv\nrules.md"));
        assert!(query.contains("中文"));
        assert!(query.contains("清洗销售数据"));
        assert!(!query.contains("{message}"));
        assert!(!query.contains("{attachments}"));
        assert!(!query.contains("{language}"));
        assert!(!query.contains("{step}"));
    }

    #[tokio::test]
    async fn execute_step_converts_message_ask_user_into_message_and_wait_events() {
        let (mut react, requests) = react(
            vec![message_ask_user_call("请提供登录验证码")],
            vec![Box::new(MessageAskUserTool::new())],
        );
        let plan = Plan {
            language: "中文".to_string(),
            ..Plan::default()
        };
        let mut step = Step::new("登录系统");

        let events = react
            .execute_step(&plan, &mut step, &Message::default())
            .await
            .unwrap();

        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], Event::Step(_)));
        let Event::Message(question) = &events[1] else {
            panic!("第二个事件必须是用户问题消息");
        };
        assert_eq!(question.message, "请提供登录验证码");
        assert!(matches!(events[2], Event::Wait(_)));
        assert_eq!(step.status, ExecutionStatus::Running);
        assert!(react
            .base
            .memory()
            .get_last_message()
            .is_some_and(|message| message.contains_key("tool_calls")));

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].tools.as_ref().map(Vec::len), Some(1));
        assert!(requests[0].tool_choice.is_none());
    }

    #[tokio::test]
    async fn execute_step_marks_failed_step_before_passing_error_event() {
        let (mut react, _) = react(vec![assistant_message(json!(123))], Vec::new());
        let plan = Plan::default();
        let mut step = Step::new("执行失败步骤");

        let events = react
            .execute_step(&plan, &mut step, &Message::default())
            .await
            .unwrap();

        assert_eq!(events.len(), 3);
        let Event::Step(failed) = &events[1] else {
            panic!("第二个事件必须是步骤失败事件");
        };
        assert_eq!(failed.status, StepEventStatus::Failed);
        assert_eq!(failed.step.status, ExecutionStatus::Failed);
        assert_eq!(
            failed.step.error.as_deref(),
            Some("Agent未能生成有效回复内容")
        );
        let Event::Error(error) = &events[2] else {
            panic!("第三个事件必须是错误事件");
        };
        assert_eq!(error.error, "Agent未能生成有效回复内容");
        assert_eq!(step.status, ExecutionStatus::Failed);
        assert_eq!(step.error.as_deref(), Some("Agent未能生成有效回复内容"));
    }

    #[tokio::test]
    async fn summarize_converts_file_paths_into_message_attachments() {
        let (mut react, requests) = react(
            vec![assistant_message(json!(
                r#"{
                    "message":"任务已完成，请查看报告。",
                    "attachments":["/tmp/report.md","/tmp/data.csv"]
                }"#
            ))],
            Vec::new(),
        );

        let events = react.summarize().await.unwrap();

        assert_eq!(events.len(), 1);
        let Event::Message(message) = &events[0] else {
            panic!("事件必须是汇总消息事件");
        };
        assert_eq!(message.role, MessageRole::Assistant);
        assert_eq!(message.message, "任务已完成，请查看报告。");
        assert_eq!(message.attachments.len(), 2);
        assert_eq!(message.attachments[0].filepath, "/tmp/report.md");
        assert_eq!(message.attachments[1].filepath, "/tmp/data.csv");

        let requests = requests.lock().unwrap();
        let query = requests[0]
            .messages
            .last()
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .unwrap();
        assert_eq!(query, SUMMARIZE_PROMPT);
    }
}
