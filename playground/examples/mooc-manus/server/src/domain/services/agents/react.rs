use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use tracing::info;

use crate::domain::{
    external::{JsonParser, Llm},
    models::{
        AgentConfig, Event, ExecutionStatus, File, Message, MessageEvent, MessageRole, Plan, Step,
        StepEvent, StepEventStatus, ToolEventStatus, WaitEvent,
    },
    repositories::SessionRepository,
    services::{
        event_sink::{EventControl, EventSink},
        prompts::{
            render_prompt, EXECUTION_PROMPT, REACT_SYSTEM_PROMPT, SUMMARIZE_PROMPT, SYSTEM_PROMPT,
        },
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
        session_id: impl Into<String>,
        session_repository: Arc<dyn SessionRepository>,
        agent_config: AgentConfig,
        llm: Arc<dyn Llm>,
        json_parser: Arc<dyn JsonParser>,
        tools: Vec<Box<dyn BaseTool>>,
    ) -> Self {
        Self {
            base: BaseAgent::new(
                react_options(),
                session_id,
                session_repository,
                agent_config,
                llm,
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
        sink: &mut dyn EventSink,
    ) -> Result<EventControl> {
        // 1. 根据传递的内容生成执行消息
        let query = render_prompt(
            EXECUTION_PROMPT,
            &[
                ("{message}", &message.message),
                ("{attachments}", &message.attachments.join("\n")),
                ("{language}", &plan.language),
                ("{step}", &step.description),
            ],
        );

        // 2. 更新步骤的执行状态为运行中并返回 Step 事件
        step.status = ExecutionStatus::Running;
        if sink
            .emit(Event::Step(StepEvent {
                step: step.clone(),
                status: StepEventStatus::Started,
                ..StepEvent::default()
            }))
            .await?
            == EventControl::Stop
        {
            return Ok(EventControl::Stop);
        }

        // 3. 调用 invoke() 获取 Agent 返回的事件内容
        let mut sink = ExecuteStepSink {
            json_parser: self.base.json_parser(),
            step: &mut *step,
            sink,
        };
        let control = self.base.invoke(&query, None, &mut sink).await?;
        if control == EventControl::Stop {
            return Ok(EventControl::Stop);
        }

        // 16. 循环迭代完成后代表子步骤已实现，需要更新状态。
        step.status = ExecutionStatus::Completed;

        Ok(EventControl::Continue)
    }

    /// 调用 Agent 汇总历史消息并生成最终回复 + 附件
    pub async fn summarize(&mut self, sink: &mut dyn EventSink) -> Result<EventControl> {
        // 1. 使用汇总 Prompt 调用 Agent 生成事件
        let mut sink = SummarizeSink {
            json_parser: self.base.json_parser(),
            sink,
        };
        self.base.invoke(SUMMARIZE_PROMPT, None, &mut sink).await
    }
}

struct ExecuteStepSink<'a> {
    json_parser: Arc<dyn JsonParser>,
    step: &'a mut Step,
    sink: &'a mut dyn EventSink,
}

#[async_trait]
impl EventSink for ExecuteStepSink<'_> {
    async fn emit(&mut self, event: Event) -> Result<EventControl> {
        // 4. 根据事件类型执行不同操作
        match event {
            // 5. 工具事件需要判断工具的名称是否为 message_ask_user
            Event::Tool(tool_event) if tool_event.function_name == "message_ask_user" => {
                // 6. 工具如果在调用中，我们需要返回一条消息告知用户需要让用户处理什么
                if tool_event.status == ToolEventStatus::Calling {
                    // message_ask_user 的问题文本记录在 text 参数中。
                    let text = tool_event
                        .function_args
                        .get("text")
                        .cloned()
                        .unwrap_or_else(|| Value::String(String::new()));
                    let message = serde_json::from_value::<String>(text)?;
                    self.sink
                        .emit(Event::Message(MessageEvent {
                            role: MessageRole::Assistant,
                            message,
                            ..MessageEvent::default()
                        }))
                        .await
                } else {
                    // 7. 如果工具事件为已调用，则需要返回等待事件并中断程序
                    self.sink.emit(Event::Wait(WaitEvent::default())).await?;
                    Ok(EventControl::Stop)
                }
            }
            Event::Message(message_event) => {
                // 8. 返回消息事件，意味着 content 有内容，则代表执行 Agent 已运行完毕
                self.step.status = ExecutionStatus::Completed;
                // 9. message 中输出的数据结构为 json，需要提取并解析
                let parsed_obj = self
                    .json_parser
                    .invoke(&message_event.message, None)
                    .await?;
                let new_step: Step = serde_json::from_value(parsed_obj)?;

                // 10. 使用结构化输出更新当前子步骤的数据
                self.step.success = new_step.success;
                self.step.result = new_step.result;
                self.step.attachments = new_step.attachments;

                // 11. 返回步骤完成事件
                if self
                    .sink
                    .emit(Event::Step(StepEvent {
                        step: self.step.clone(),
                        status: StepEventStatus::Completed,
                        ..StepEvent::default()
                    }))
                    .await?
                    == EventControl::Stop
                {
                    return Ok(EventControl::Stop);
                }

                // 12. 子步骤存在结果时，将结果消息返回给用户
                if let Some(result) = self
                    .step
                    .result
                    .as_deref()
                    .filter(|result| !result.is_empty())
                {
                    return self
                        .sink
                        .emit(Event::Message(MessageEvent {
                            role: MessageRole::Assistant,
                            message: result.to_owned(),
                            ..MessageEvent::default()
                        }))
                        .await;
                }
                Ok(EventControl::Continue)
            }
            Event::Error(error_event) => {
                // 13. 错误事件更新步骤状态和错误信息
                self.step.status = ExecutionStatus::Failed;
                self.step.error = Some(error_event.error.clone());

                // 14. 返回子步骤对应事件
                if self
                    .sink
                    .emit(Event::Step(StepEvent {
                        step: self.step.clone(),
                        status: StepEventStatus::Failed,
                        ..StepEvent::default()
                    }))
                    .await?
                    == EventControl::Stop
                {
                    return Ok(EventControl::Stop);
                }
                self.sink.emit(Event::Error(error_event)).await
            }
            // 15. 其他事件直接返回
            event => self.sink.emit(event).await,
        }
    }
}

struct SummarizeSink<'a> {
    json_parser: Arc<dyn JsonParser>,
    sink: &'a mut dyn EventSink,
}

#[async_trait]
impl EventSink for SummarizeSink<'_> {
    async fn emit(&mut self, event: Event) -> Result<EventControl> {
        match event {
            // 2. MessageEvent 表示 Agent 已经生成结构化汇总内容
            Event::Message(message_event) => {
                // 3. 记录日志并解析输出内容
                info!(message = %message_event.message, "执行Agent生成汇总内容");
                let parsed_obj = self
                    .json_parser
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
                self.sink
                    .emit(Event::Message(MessageEvent {
                        role: MessageRole::Assistant,
                        message: message.message,
                        attachments,
                        ..MessageEvent::default()
                    }))
                    .await
            }
            // 6. 其他事件直接返回
            event => self.sink.emit(event).await,
        }
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
        system_prompt: format!("{SYSTEM_PROMPT}{REACT_SYSTEM_PROMPT}"),
        // format 控制的是 content，工具调用控制的是 tool_calls，两者可以同时使用。
        format: Some("json_object".to_string()),
        ..AgentOptions::default()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use tokio::sync::oneshot;

    use super::*;
    use crate::domain::{
        external::{LlmMessage, Response, ResponseFormat, Tool, ToolChoice},
        models::ToolResult,
        services::{
            agents::test_support::MemoryRepository,
            tools::{MessageTool, ToolArguments, ToolDefinition},
        },
    };

    impl ReActAgent {
        pub(crate) async fn execute_step_collect(
            &mut self,
            plan: &Plan,
            step: &mut Step,
            message: &Message,
        ) -> Result<Vec<Event>> {
            let mut sink =
                crate::domain::services::agents::test_support::CollectedEvents::default();
            self.execute_step(plan, step, message, &mut sink).await?;
            Ok(sink.events)
        }

        pub(crate) async fn summarize_collect(&mut self) -> Result<Vec<Event>> {
            let mut sink =
                crate::domain::services::agents::test_support::CollectedEvents::default();
            self.summarize(&mut sink).await?;
            Ok(sink.events)
        }
    }

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

    fn react(
        responses: Vec<Response>,
        tools: Vec<Box<dyn BaseTool>>,
    ) -> (ReActAgent, Requests, Arc<MemoryRepository>) {
        react_with_iterations(responses, tools, 3)
    }

    fn react_with_iterations(
        responses: Vec<Response>,
        tools: Vec<Box<dyn BaseTool>>,
        max_iterations: i64,
    ) -> (ReActAgent, Requests, Arc<MemoryRepository>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let llm = MockLlm {
            responses: Mutex::new(VecDeque::from(responses)),
            requests: Arc::clone(&requests),
        };
        let repository = Arc::new(MemoryRepository::default());
        let react = ReActAgent::new(
            "session-1",
            repository.clone(),
            AgentConfig {
                max_iterations,
                max_retries: 1,
                max_search_results: 10,
            },
            Arc::new(llm),
            Arc::new(MockJsonParser),
            tools,
        );
        (react, requests, repository)
    }

    #[derive(Default)]
    struct StopSink {
        events: Vec<Event>,
        stop_after: usize,
    }

    #[async_trait]
    impl EventSink for StopSink {
        async fn emit(&mut self, event: Event) -> Result<EventControl> {
            self.events.push(event);
            Ok(if self.events.len() == self.stop_after {
                EventControl::Stop
            } else {
                EventControl::Continue
            })
        }
    }

    struct StartedSink {
        events: Vec<Event>,
        entered: Option<oneshot::Sender<()>>,
        release: Option<oneshot::Receiver<()>>,
    }

    #[async_trait]
    impl EventSink for StartedSink {
        async fn emit(&mut self, event: Event) -> Result<EventControl> {
            let started =
                matches!(&event, Event::Step(step) if step.status == StepEventStatus::Started);
            self.events.push(event);
            if started {
                self.entered.take().unwrap().send(()).unwrap();
                self.release.take().unwrap().await.unwrap();
            }
            Ok(EventControl::Continue)
        }
    }

    #[tokio::test]
    async fn started_event_delivery_precedes_first_model_request() {
        let (mut react, requests, repository) = react(
            vec![assistant_message(json!(
                r#"{"success":true,"result":"done"}"#
            ))],
            Vec::new(),
        );
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            let mut step = Step::new("run");
            let mut sink = StartedSink {
                events: Vec::new(),
                entered: Some(entered_tx),
                release: Some(release_rx),
            };
            let control = react
                .execute_step(&Plan::default(), &mut step, &Message::default(), &mut sink)
                .await
                .unwrap();
            (control, sink.events)
        });

        entered_rx.await.unwrap();
        assert!(requests.lock().unwrap().is_empty());
        assert_eq!(
            repository.writes.load(std::sync::atomic::Ordering::SeqCst),
            0
        );

        release_tx.send(()).unwrap();
        let (control, events) = task.await.unwrap();
        assert_eq!(control, EventControl::Continue);
        assert_eq!(events.len(), 3);
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn stop_on_each_step_result_event_prevents_later_events() {
        for stop_after in 1..=3 {
            let (mut react, requests, _) = react(
                vec![assistant_message(json!(
                    r#"{"success":true,"result":"done"}"#
                ))],
                Vec::new(),
            );
            let mut step = Step::new("run");
            let mut sink = StopSink {
                events: Vec::new(),
                stop_after,
            };

            let control = react
                .execute_step(&Plan::default(), &mut step, &Message::default(), &mut sink)
                .await
                .unwrap();

            assert_eq!(control, EventControl::Stop);
            assert_eq!(sink.events.len(), stop_after);
            assert_eq!(requests.lock().unwrap().len(), usize::from(stop_after > 1));
            assert_eq!(
                step.status,
                if stop_after == 1 {
                    ExecutionStatus::Running
                } else {
                    ExecutionStatus::Completed
                }
            );
        }
    }

    #[tokio::test]
    async fn each_error_derivative_can_stop_before_the_later_message() {
        for stop_after in [2, 3] {
            let (mut react, requests, _) = react_with_iterations(
                vec![assistant_message(json!(
                    r#"{"success":true,"result":"late"}"#
                ))],
                Vec::new(),
                0,
            );
            let mut step = Step::new("run");
            let mut sink = StopSink {
                events: Vec::new(),
                stop_after,
            };

            let control = react
                .execute_step(&Plan::default(), &mut step, &Message::default(), &mut sink)
                .await
                .unwrap();

            assert_eq!(control, EventControl::Stop);
            assert_eq!(sink.events.len(), stop_after);
            assert!(
                matches!(&sink.events[1], Event::Step(event) if event.status == StepEventStatus::Failed)
            );
            assert_eq!(step.status, ExecutionStatus::Failed);
            assert!(step.error.as_deref().unwrap().contains("最大迭代次数"));
            assert!(step.result.is_none());
            assert_eq!(requests.lock().unwrap().len(), 1);
        }
    }

    #[tokio::test]
    async fn iteration_limit_keeps_existing_error_then_message_order() {
        let (mut react, _, _) = react_with_iterations(
            vec![assistant_message(json!(
                r#"{"success":true,"result":"late"}"#
            ))],
            Vec::new(),
            0,
        );
        let mut step = Step::new("run");
        let events = react
            .execute_step_collect(&Plan::default(), &mut step, &Message::default())
            .await
            .unwrap();

        assert_eq!(events.len(), 5);
        assert!(
            matches!(&events[1], Event::Step(event) if event.status == StepEventStatus::Failed)
        );
        assert!(matches!(&events[2], Event::Error(_)));
        assert!(
            matches!(&events[3], Event::Step(event) if event.status == StepEventStatus::Completed)
        );
        assert!(matches!(&events[4], Event::Message(event) if event.message == "late"));
        assert_eq!(step.status, ExecutionStatus::Completed);
    }

    #[tokio::test]
    async fn parse_failure_keeps_the_started_event() {
        let (mut react, _, _) = react(vec![assistant_message(json!("broken json"))], Vec::new());
        let mut step = Step::new("run");
        let mut sink = crate::domain::services::agents::test_support::CollectedEvents::default();

        assert!(react
            .execute_step(&Plan::default(), &mut step, &Message::default(), &mut sink)
            .await
            .is_err());

        assert_eq!(sink.events.len(), 1);
        assert!(
            matches!(&sink.events[0], Event::Step(event) if event.status == StepEventStatus::Started)
        );
    }

    #[tokio::test]
    async fn ask_user_wait_stops_with_a_continuing_sink_and_keeps_pending_memory() {
        let (mut react, requests, repository) = react(
            vec![message_ask_user_call("验证码")],
            vec![Box::new(MessageTool::new())],
        );
        let mut step = Step::new("login");
        let mut sink = crate::domain::services::agents::test_support::CollectedEvents::default();

        let control = react
            .execute_step(&Plan::default(), &mut step, &Message::default(), &mut sink)
            .await
            .unwrap();

        assert_eq!(control, EventControl::Stop);
        assert_eq!(sink.events.len(), 3);
        assert!(matches!(&sink.events[1], Event::Message(event) if event.message == "验证码"));
        assert!(matches!(&sink.events[2], Event::Wait(_)));
        assert_eq!(requests.lock().unwrap().len(), 1);
        assert!(repository
            .memory("session-1", "react")
            .get_last_message()
            .is_some_and(|message| message.contains_key("tool_calls")));
        assert_eq!(step.status, ExecutionStatus::Running);
    }

    #[tokio::test]
    async fn summarized_message_propagates_stop() {
        let (mut react, _, _) = react(
            vec![assistant_message(json!(
                r#"{"message":"done","attachments":[]}"#
            ))],
            Vec::new(),
        );
        let mut sink = StopSink {
            events: Vec::new(),
            stop_after: 1,
        };

        assert_eq!(
            react.summarize(&mut sink).await.unwrap(),
            EventControl::Stop
        );
        assert!(matches!(&sink.events[0], Event::Message(message) if message.message == "done"));
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
        let (mut react, requests, _) = react(
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
            .execute_step_collect(&plan, &mut step, &message)
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
        assert!(request.tools.as_ref().is_some_and(Vec::is_empty));
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
        let (mut react, requests, repository) = react(
            vec![message_ask_user_call("请提供登录验证码")],
            vec![Box::new(MessageTool::new())],
        );
        let plan = Plan {
            language: "中文".to_string(),
            ..Plan::default()
        };
        let mut step = Step::new("登录系统");

        let events = react
            .execute_step_collect(&plan, &mut step, &Message::default())
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
        assert!(repository
            .memory("session-1", "react")
            .get_last_message()
            .is_some_and(|message| message.contains_key("tool_calls")));

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].tools.as_ref().map(Vec::len), Some(2));
        assert!(requests[0].tool_choice.is_none());
    }

    struct CountingAskTool {
        tool: MessageTool,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl BaseTool for CountingAskTool {
        fn name(&self) -> &str {
            "message"
        }
        fn tool_definitions(&self) -> &[ToolDefinition] {
            self.tool.tool_definitions()
        }
        async fn call_tool(
            &self,
            _tool_name: &str,
            _kwargs: ToolArguments,
        ) -> Result<ToolResult<Value>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult::default())
        }
    }

    #[tokio::test]
    async fn ask_user_missing_text_defaults_to_empty_and_invalid_values_stop_before_tool_execution()
    {
        for arguments in [
            json!({}),
            json!({"text": null}),
            json!({"text": 1}),
            json!({"text": []}),
        ] {
            let mut response = message_ask_user_call("question");
            response["tool_calls"][0]["function"]["arguments"] = json!(arguments.to_string());
            let calls = Arc::new(AtomicUsize::new(0));
            let (mut react, _, repository) = react(
                vec![response],
                vec![Box::new(CountingAskTool {
                    tool: MessageTool::new(),
                    calls: calls.clone(),
                })],
            );
            let mut step = Step::new("询问用户");
            let mut sink =
                crate::domain::services::agents::test_support::CollectedEvents::default();

            let result = react
                .execute_step(&Plan::default(), &mut step, &Message::default(), &mut sink)
                .await;

            if arguments.get("text").is_none() {
                assert_eq!(result.unwrap(), EventControl::Stop);
                assert!(
                    matches!(&sink.events[1], Event::Message(event) if event.message.is_empty())
                );
                assert!(matches!(&sink.events[2], Event::Wait(_)));
                assert_eq!(calls.load(Ordering::SeqCst), 1);
            } else {
                assert!(result
                    .unwrap_err()
                    .to_string()
                    .contains("expected a string"));
                assert!(
                    matches!(sink.events.as_slice(), [Event::Step(event)] if event.status == StepEventStatus::Started)
                );
                assert_eq!(calls.load(Ordering::SeqCst), 0);
            }
            assert_eq!(step.status, ExecutionStatus::Running);
            assert!(repository
                .memory("session-1", "react")
                .get_last_message()
                .unwrap()
                .contains_key("tool_calls"));
        }
    }

    #[tokio::test]
    async fn execute_step_marks_failed_step_before_passing_error_event() {
        let (mut react, _, _) = react(
            vec![LlmMessage::from_iter([("role".to_string(), json!("tool"))])],
            Vec::new(),
        );
        let plan = Plan::default();
        let mut step = Step::new("执行失败步骤");

        let events = react
            .execute_step_collect(&plan, &mut step, &Message::default())
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
        // 已交付的 Failed 事件保留当时快照；自然结束后步骤状态按流程更新为 Completed。
        assert_eq!(step.status, ExecutionStatus::Completed);
        assert_eq!(step.error.as_deref(), Some("Agent未能生成有效回复内容"));
    }

    #[tokio::test]
    async fn summarize_converts_file_paths_into_message_attachments() {
        let (mut react, requests, _) = react(
            vec![assistant_message(json!(
                r#"{
                    "message":"任务已完成，请查看报告。",
                    "attachments":["/tmp/report.md","/tmp/data.csv"]
                }"#
            ))],
            Vec::new(),
        );

        let events = react.summarize_collect().await.unwrap();

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

    #[tokio::test]
    async fn execute_step_preserves_template_like_user_text_and_attachment_paths() {
        let (mut react, requests, _) = react(
            vec![assistant_message(json!(
                r#"{"success":true,"result":"","attachments":[]}"#
            ))],
            Vec::new(),
        );
        let message = Message {
            message: "请解释 {attachments}、{language} 和 {step}".into(),
            attachments: vec!["/home/ubuntu/{step}.md".into()],
        };
        let plan = Plan {
            language: "中文".into(),
            ..Plan::default()
        };
        let mut step = Step::new("解释 {message}");
        react
            .execute_step_collect(&plan, &mut step, &message)
            .await
            .unwrap();
        let requests = requests.lock().unwrap();
        let query = requests[0].messages.last().unwrap()["content"]
            .as_str()
            .unwrap();
        assert!(query.contains(&message.message));
        assert!(query.contains(&message.attachments[0]));
        assert!(query.contains(&step.description));
    }
}
