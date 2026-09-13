use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use tracing::{debug, info};

use crate::domain::{
    external::{Browser, JsonParser, Llm, Sandbox, SearchEngine},
    models::{
        AgentConfig, DoneEvent, Event, ExecutionStatus, Message, MessageEvent, MessageRole, Plan,
        PlanEvent, PlanEventStatus, SessionStatus, Step, TitleEvent,
    },
    repositories::SessionRepository,
    services::{
        agents::{Agent, PlannerAgent, ReActAgent},
        tools::{
            A2ATool, BaseTool, BrowserTool, FileTool, McpTool, MessageTool, SearchTool, ShellTool,
        },
    },
};

use super::{BaseFlow, FlowStatus};

/// 规划与执行流：使用状态机协调规划 Agent 和执行 Agent。
pub struct PlannerReActFlow {
    /// 当前会话 id
    session_id: String,
    /// 会话仓库
    session_repository: Arc<dyn SessionRepository>,
    /// 当前流状态
    status: FlowStatus,
    /// 当前计划；首次规划前为空，恢复会话时从历史事件读取
    plan: Option<Plan>,
    /// 规划 Agent
    planner: PlannerAgent,
    /// 执行 Agent
    react: ReActAgent,
}

impl PlannerReActFlow {
    /// 构造函数，完成规划与执行流及其工具、Agent 的初始化。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        llm: Arc<dyn Llm>,
        agent_config: AgentConfig,
        session_id: impl Into<String>,
        session_repository: Arc<dyn SessionRepository>,
        json_parser: Arc<dyn JsonParser>,
        browser: Box<dyn Browser>,
        sandbox: Arc<dyn Sandbox>,
        search_engine: Box<dyn SearchEngine>,
        mcp_tool: McpTool,
        a2a_tool: A2ATool,
    ) -> Self {
        let session_id = session_id.into();

        // 1. 初始化 Agent 预设工具列表。沙箱由文件和 Shell 工具共同持有。
        let tools: Vec<Box<dyn BaseTool>> = vec![
            Box::new(FileTool::new(sandbox.clone())),
            Box::new(ShellTool::new(sandbox)),
            Box::new(BrowserTool::new(browser)),
            Box::new(SearchTool::new(search_engine)),
            Box::new(MessageTool::new()),
            Box::new(mcp_tool),
            Box::new(a2a_tool),
        ];

        // 2. 创建规划 Agent。规划只负责生成结构化计划，不暴露工具。
        let planner = PlannerAgent::new(
            session_id.clone(),
            session_repository.clone(),
            agent_config.clone(),
            llm.clone(),
            json_parser.clone(),
        );
        debug!(session_id, "创建规划Agent成功");

        // 3. 创建执行 Agent，并把完整工具集合交给它。
        let react = ReActAgent::new(
            session_id.clone(),
            session_repository.clone(),
            agent_config,
            llm,
            json_parser,
            tools,
        );
        debug!(session_id, "创建执行Agent成功");

        Self {
            session_id,
            session_repository,
            status: FlowStatus::Idle,
            plan: None,
            planner,
            react,
        }
    }

    /// 使用已经组装好的两个 Agent 创建测试流。
    #[cfg(test)]
    fn from_agents(
        session_id: impl Into<String>,
        session_repository: Arc<dyn SessionRepository>,
        planner: PlannerAgent,
        react: ReActAgent,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            session_repository,
            status: FlowStatus::Idle,
            plan: None,
            planner,
            react,
        }
    }

    /// 返回当前流状态。
    pub const fn status(&self) -> FlowStatus {
        self.status
    }

    /// 返回当前计划。
    pub fn plan(&self) -> Option<&Plan> {
        self.plan.as_ref()
    }

    /// 初始化执行 Agent 独占的工具，确保后续调用使用同一实例。
    pub(crate) async fn initialize_tools(&mut self) -> Result<()> {
        self.react.base_mut().initialize_tools().await
    }

    /// 按任务运行器的顺序清理远程工具资源。
    pub(crate) async fn cleanup_tools(&mut self) -> Result<()> {
        // 2.清除 mcp 工具
        info!("销毁AgentTaskRunner中的mcp工具");
        self.react.base_mut().cleanup_tools(&["mcp"]).await?;

        // 3.清除 a2a 工具
        info!("销毁AgentTaskRunner中的a2a工具");
        self.react.base_mut().cleanup_tools(&["a2a"]).await
    }
}

#[async_trait]
impl BaseFlow for PlannerReActFlow {
    /// 传递消息运行流，在流中调用 Planner 和 ReAct Agent 组合完成任务并返回对应事件。
    async fn invoke(&mut self, message: Message) -> Result<Vec<Event>> {
        // 1. 调用会话仓库查询会话是否存在
        let session = self
            .session_repository
            .get_by_id(&self.session_id)
            .await?
            .ok_or_else(|| anyhow!("会话[{}]不存在，请核实后尝试", self.session_id))?;

        // 2. 非空闲会话可能仍在运行，也可能正在等待人类输入。
        // 两种情况都要先闭合或移除末尾未完成的工具调用，保证消息序列合法。
        if session.status != SessionStatus::Pending {
            debug!(
                session_id = self.session_id,
                "会话未处于空闲状态，回滚Agent记忆"
            );
            self.planner.roll_back(message.clone()).await?;
            self.react.roll_back(message.clone()).await?;
        }

        // 3. 运行中收到新消息，需要重新规划。
        if session.status == SessionStatus::Running {
            debug!(
                session_id = self.session_id,
                "运行中的会话收到新消息，重新规划"
            );
            self.status = FlowStatus::Planning;
        }

        // 4. 等待状态收到人类回复，从执行阶段继续。
        if session.status == SessionStatus::Waiting {
            debug!(
                session_id = self.session_id,
                "等待中的会话收到人类回复，继续执行"
            );
            self.status = FlowStatus::Executing;
        }

        // 5. 流开始工作后，会话统一标记为运行中。
        self.session_repository
            .update_status(&self.session_id, SessionStatus::Running)
            .await?;

        // 6. 从会话历史恢复最新计划。
        self.plan = session.get_latest_plan().cloned();
        info!(
            session_id = self.session_id,
            message = %message.message.chars().take(50).collect::<String>(),
            "Planner&ReAct流接收消息"
        );

        let mut output = Vec::new();
        // 更新计划发生在下一次状态循环中，因此保留刚执行步骤的快照。
        // 这里持有 Step 的所有权，避免跨 `.await` 保存对 Plan 内部字段的借用。
        let mut current_step: Option<Step> = None;

        // 7. 使用状态机控制规划、执行、更新、总结和完成的完整生命周期。
        loop {
            match self.status {
                // 8. 空闲状态只负责进入规划阶段。
                FlowStatus::Idle => {
                    info!("Planner&ReAct流状态从idle变成planning");
                    self.status = FlowStatus::Planning;
                }
                // 9. 规划阶段调用规划 Agent 创建计划。
                FlowStatus::Planning => {
                    info!("Planner&ReAct流开始创建计划");
                    for event in self.planner.create_plan(message.clone()).await? {
                        // 10. 创建成功时更新当前计划，并派生标题和初始 AI 消息事件。
                        if let Event::Plan(plan_event) = &event {
                            if plan_event.status == PlanEventStatus::Created {
                                self.plan = Some(plan_event.plan.clone());
                                info!(
                                    steps = plan_event.plan.steps.len(),
                                    "Planner&ReAct流成功创建计划"
                                );
                                output.push(Event::Title(TitleEvent {
                                    title: plan_event.plan.title.clone(),
                                    ..TitleEvent::default()
                                }));
                                output.push(Event::Message(MessageEvent {
                                    role: MessageRole::Assistant,
                                    message: plan_event.plan.message.clone(),
                                    ..MessageEvent::default()
                                }));
                            }
                        }
                        output.push(event);
                    }

                    // 11. 计划创建后进入执行阶段；无计划或无步骤则直接标记完成。
                    self.status = FlowStatus::Executing;
                    if self
                        .plan
                        .as_ref()
                        .map_or(true, |plan| plan.steps.is_empty())
                    {
                        info!("Planner&ReAct流创建计划失败或无子步骤");
                        self.status = FlowStatus::Completed;
                    }
                }
                // 12. 执行阶段每次取出一个尚未结束的步骤交给 ReAct Agent。
                FlowStatus::Executing => {
                    let plan = self
                        .plan
                        .as_mut()
                        .ok_or_else(|| anyhow!("规划与执行流缺少可执行计划"))?;
                    plan.status = ExecutionStatus::Running;

                    let Some(step_index) = plan.steps.iter().position(|step| !step.done()) else {
                        info!("Planner&ReAct流状态从executing变成summarizing");
                        self.status = FlowStatus::Summarizing;
                        continue;
                    };

                    // Rust 不能同时可变借用步骤并不可变借用整个计划，
                    // 因此创建只读快照作为执行上下文，再单独修改原计划中的步骤。
                    let plan_context = plan.clone();
                    let step = &mut plan.steps[step_index];
                    info!(step_id = step.id, description = %step.description.chars().take(50).collect::<String>(), "Planner&ReAct流开始执行步骤");
                    let step_events = self
                        .react
                        .execute_step(&plan_context, step, &message)
                        .await?;
                    let waiting_for_user = step_events
                        .iter()
                        .any(|event| matches!(event, Event::Wait(_)));
                    output.extend(step_events);

                    // Vec<Event> 无法像异步事件流一样停在一次 yield 上。
                    // 遇到 WaitEvent 时主动结束本轮调用，等待用户回复后从 Executing 恢复。
                    if waiting_for_user {
                        info!("Planner&ReAct流等待用户回复");
                        return Ok(output);
                    }
                    current_step = Some(step.clone());

                    // 13. 步骤结束后压缩工具结果，控制上下文长度。
                    info!("压缩react Agent记忆/上下文");
                    self.react.compact_memory().await?;
                    self.status = FlowStatus::Updating;
                }
                FlowStatus::Updating => {
                    // 23. 流状态为更新，表示需要根据刚执行的步骤调整后续计划。
                    info!("Planner&ReAct流开始更新计划");
                    let step = current_step
                        .as_ref()
                        .ok_or_else(|| anyhow!("规划与执行流缺少待更新步骤"))?;
                    let plan = self
                        .plan
                        .as_mut()
                        .ok_or_else(|| anyhow!("规划与执行流缺少可更新计划"))?;
                    output.extend(self.planner.update_plan(plan, step).await?);

                    // 24. 更新完成后重新进入执行状态，读取新的首个未完成步骤。
                    info!("Planner&ReAct流状态从updating变成executing");
                    self.status = FlowStatus::Executing;
                }
                FlowStatus::Summarizing => {
                    // 25. 流状态为总结中，表示全部子步骤已经执行完成。
                    info!("Planner&ReAct流开始总结");
                    output.extend(self.react.summarize().await?);

                    // 26. 总结完成，流进入完成状态。
                    info!("Planner&ReAct流状态从summarizing变成completed");
                    self.status = FlowStatus::Completed;
                }
                FlowStatus::Completed => {
                    // 27. 同步计划状态，并发送完成事件通知上层应用。
                    if let Some(plan) = self.plan.as_mut() {
                        plan.status = ExecutionStatus::Completed;
                        output.push(Event::Plan(PlanEvent {
                            plan: plan.clone(),
                            status: PlanEventStatus::Completed,
                            ..PlanEvent::default()
                        }));
                    }

                    // 本次任务已经结束，回到空闲状态并跳出控制循环。
                    self.status = FlowStatus::Idle;
                    break;
                }
            }
        }

        // 28. DoneEvent 是整条事件流的结束标记，必须在退出控制循环后发送。
        output.push(Event::Done(DoneEvent::default()));
        info!("Planner&ReAct流处理任务消息已完毕");
        Ok(output)
    }

    fn done(&self) -> bool {
        self.status == FlowStatus::Idle
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
        time::Duration,
    };

    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use tokio::time::timeout;

    use super::*;
    use crate::domain::{
        external::{LlmMessage, Response, ResponseFormat, Tool, ToolChoice},
        models::{Memory, PlanEvent, Session, Step},
        services::agents::test_support::MemoryRepository,
    };

    type Requests = Arc<Mutex<Vec<Vec<LlmMessage>>>>;

    struct MockLlm {
        responses: Mutex<VecDeque<Response>>,
        requests: Requests,
    }

    #[async_trait]
    impl Llm for MockLlm {
        async fn invoke(
            &self,
            messages: Vec<LlmMessage>,
            _tools: Option<Vec<Tool>>,
            _response_format: Option<ResponseFormat>,
            _tool_choice: Option<ToolChoice>,
        ) -> Result<Response> {
            self.requests.lock().unwrap().push(messages);
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

    fn assistant_json(value: Value) -> Response {
        Response::from_iter([
            ("role".to_string(), json!("assistant")),
            ("content".to_string(), Value::String(value.to_string())),
        ])
    }

    fn plan_response(steps: Vec<Step>) -> Response {
        assistant_json(json!({
            "id": "plan-1",
            "title": "整理项目",
            "goal": "完成项目整理",
            "language": "中文",
            "message": "我会先检查，再整理结果。",
            "steps": steps,
        }))
    }

    fn step_response(result: &str) -> Response {
        assistant_json(json!({
            "status": "completed",
            "result": result,
            "success": true,
        }))
    }

    fn summary_response(message: &str) -> Response {
        assistant_json(json!({
            "message": message,
            "attachments": [],
        }))
    }

    fn ask_user_tool_call() -> LlmMessage {
        LlmMessage::from_iter([
            ("role".to_string(), json!("assistant")),
            ("content".to_string(), Value::Null),
            (
                "tool_calls".to_string(),
                json!([{
                    "id": "call-1",
                    "function": {
                        "name": "message_ask_user",
                        "arguments": "{\"text\":\"是否继续？\"}"
                    }
                }]),
            ),
        ])
    }

    fn flow(
        repository: Arc<MemoryRepository>,
        responses: Vec<Response>,
    ) -> (PlannerReActFlow, Requests) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let llm: Arc<dyn Llm> = Arc::new(MockLlm {
            responses: Mutex::new(VecDeque::from(responses)),
            requests: requests.clone(),
        });
        let json_parser: Arc<dyn JsonParser> = Arc::new(MockJsonParser);
        let config = AgentConfig {
            max_iterations: 3,
            max_retries: 1,
            max_search_results: 10,
        };
        let planner = PlannerAgent::new(
            "session-1",
            repository.clone(),
            config.clone(),
            llm.clone(),
            json_parser.clone(),
        );
        let react = ReActAgent::new(
            "session-1",
            repository.clone(),
            config,
            llm,
            json_parser,
            vec![Box::new(MessageTool::new())],
        );

        (
            PlannerReActFlow::from_agents("session-1", repository, planner, react),
            requests,
        )
    }

    #[tokio::test]
    async fn rejects_a_missing_session_before_calling_llm() {
        let repository = Arc::new(MemoryRepository::default());
        let (mut flow, requests) = flow(repository, Vec::new());

        let error = flow.invoke(Message::default()).await.unwrap_err();

        assert!(error.to_string().contains("会话[session-1]不存在"));
        assert!(requests.lock().unwrap().is_empty());
        assert!(flow.done());
    }

    #[tokio::test]
    async fn runs_update_summary_and_completion_to_idle() {
        let repository = Arc::new(MemoryRepository::default());
        repository.insert_session(Session {
            id: "session-1".to_string(),
            ..Session::default()
        });
        let (mut flow, requests) = flow(
            repository.clone(),
            vec![
                plan_response(vec![Step::new("检查项目结构")]),
                step_response("检查完成"),
                plan_response(Vec::new()),
                summary_response("项目整理完成。"),
            ],
        );

        let events = timeout(
            Duration::from_secs(1),
            flow.invoke(Message {
                message: "帮我整理项目".to_string(),
                attachments: Vec::new(),
            }),
        )
        .await
        .expect("完整状态机应在完成后退出")
        .unwrap();

        assert_eq!(flow.status(), FlowStatus::Idle);
        assert!(flow.done());
        assert_eq!(events.len(), 10);
        assert!(matches!(events[0], Event::Title(_)));
        assert!(matches!(events[1], Event::Message(_)));
        assert!(matches!(
            events[2],
            Event::Plan(PlanEvent {
                status: PlanEventStatus::Created,
                ..
            })
        ));
        assert!(matches!(events[3], Event::Step(_)));
        assert!(matches!(events[4], Event::Step(_)));
        assert!(matches!(events[5], Event::Message(_)));
        assert!(matches!(
            events[6],
            Event::Plan(PlanEvent {
                status: PlanEventStatus::Updated,
                ..
            })
        ));
        assert!(matches!(events[7], Event::Message(_)));
        assert!(matches!(
            events[8],
            Event::Plan(PlanEvent {
                status: PlanEventStatus::Completed,
                ..
            })
        ));
        assert!(matches!(events[9], Event::Done(_)));

        let plan = flow.plan().unwrap();
        assert_eq!(plan.status, ExecutionStatus::Completed);
        assert_eq!(plan.steps[0].status, ExecutionStatus::Completed);
        assert_eq!(plan.steps[0].result.as_deref(), Some("检查完成"));
        assert_eq!(
            repository.session("session-1").unwrap().status,
            SessionStatus::Running
        );
        assert_eq!(requests.lock().unwrap().len(), 4);
    }

    #[tokio::test]
    async fn wait_event_pauses_before_updating_the_plan() {
        let repository = Arc::new(MemoryRepository::default());
        repository.insert_session(Session {
            id: "session-1".to_string(),
            ..Session::default()
        });
        let (mut flow, requests) = flow(
            repository,
            vec![
                plan_response(vec![Step::new("等待用户确认")]),
                ask_user_tool_call(),
            ],
        );

        let events = flow
            .invoke(Message {
                message: "执行前请确认".to_string(),
                attachments: Vec::new(),
            })
            .await
            .unwrap();

        assert_eq!(flow.status(), FlowStatus::Executing);
        assert!(!flow.done());
        assert_eq!(requests.lock().unwrap().len(), 2);
        assert!(matches!(events.last(), Some(Event::Wait(_))));
        assert!(!events.iter().any(|event| matches!(event, Event::Done(_))));
        assert!(!events.iter().any(|event| {
            matches!(
                event,
                Event::Plan(PlanEvent {
                    status: PlanEventStatus::Updated | PlanEventStatus::Completed,
                    ..
                })
            )
        }));
        assert_eq!(
            flow.plan().unwrap().steps[0].status,
            ExecutionStatus::Running
        );
    }

    #[tokio::test]
    async fn empty_plan_completes_without_running_react() {
        let repository = Arc::new(MemoryRepository::default());
        repository.insert_session(Session {
            id: "session-1".to_string(),
            ..Session::default()
        });
        let (mut flow, requests) = flow(repository, vec![plan_response(Vec::new())]);

        let events = flow.invoke(Message::default()).await.unwrap();

        assert_eq!(flow.status(), FlowStatus::Idle);
        assert!(flow.done());
        assert_eq!(requests.lock().unwrap().len(), 1);
        assert_eq!(events.len(), 5);
        let Event::Plan(PlanEvent { plan, .. }) = &events[2] else {
            panic!("第三个事件必须是计划事件");
        };
        assert!(plan.steps.is_empty());
        assert!(matches!(
            events[3],
            Event::Plan(PlanEvent {
                status: PlanEventStatus::Completed,
                ..
            })
        ));
        assert!(matches!(events[4], Event::Done(_)));
    }

    #[tokio::test]
    async fn waiting_session_resumes_from_latest_plan() {
        let repository = Arc::new(MemoryRepository::default());
        let mut react_memory = Memory::new();
        react_memory.add_messages(vec![
            LlmMessage::from_iter([
                ("role".to_string(), json!("system")),
                ("content".to_string(), json!("system prompt")),
            ]),
            ask_user_tool_call(),
        ]);
        repository.insert("session-1", "react", react_memory);
        let plan = Plan {
            id: "plan-1".to_string(),
            language: "中文".to_string(),
            steps: vec![Step::new("继续执行")],
            ..Plan::default()
        };
        repository.insert_session(Session {
            id: "session-1".to_string(),
            status: SessionStatus::Waiting,
            events: vec![Event::Plan(PlanEvent {
                plan,
                status: PlanEventStatus::Created,
                ..PlanEvent::default()
            })],
            ..Session::default()
        });
        let (mut flow, requests) = flow(
            repository,
            vec![
                step_response("继续完成"),
                plan_response(Vec::new()),
                summary_response("继续执行的任务已完成。"),
            ],
        );

        let events = flow
            .invoke(Message {
                message: "继续".to_string(),
                attachments: Vec::new(),
            })
            .await
            .unwrap();

        assert_eq!(flow.status(), FlowStatus::Idle);
        assert!(flow.done());
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(Memory::get_message_role(&requests[0][2]), Some("tool"));
        assert!(requests[0][2]
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("继续")));
        assert!(events.iter().any(|event| matches!(event, Event::Step(_))));
        assert_eq!(
            flow.plan().unwrap().steps[0].result.as_deref(),
            Some("继续完成")
        );
        assert_eq!(flow.plan().unwrap().status, ExecutionStatus::Completed);
        assert!(matches!(events.last(), Some(Event::Done(_))));
    }
}
