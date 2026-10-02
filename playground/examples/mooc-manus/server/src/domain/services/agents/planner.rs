//! 多 Agent 系统/flow = PlannerAgent + ReActAgent
//!
//! 顺序：
//!     1. PlannerAgent 生成规划；
//!     2. 循环取出规划中的子步骤，让 ReActAgent 执行，依次迭代；
//!     3. ReActAgent 执行完每一个子步骤之后，需要将子步骤结果 + Plan 传递给 PlannerAgent 让其更新计划/Plan；
//!     4. 循环取出规划中的子步骤，让 ReActAgent 执行，依次迭代；
//!     5. ...
//!     6. 直到所有子任务/子步骤都完成，这时候将子步骤的所有结果汇总进行总结(ReActAgent)；
//!
//! PlannerAgent：
//! - 功能：将用户的需求拆解成多个子任务 + 根据已完成的子任务更新规划
//! - 提示词：创建规划的 prompt、更新规划的 prompt
//!
//! ReActAgent：
//! - 功能：迭代执行完每一个子任务、汇总所有的子任务进行总结
//! - 提示词：执行任务的 prompt、汇总总结 prompt
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tracing::info;

use crate::domain::{
    external::{JsonParser, Llm},
    models::{AgentConfig, Event, Message, Plan, PlanEvent, PlanEventStatus, Step},
    repositories::SessionRepository,
    services::{
        event_sink::{EventControl, EventSink},
        prompts::{
            render_prompt, CREATE_PLAN_PROMPT, PLANNER_SYSTEM_PROMPT, SYSTEM_PROMPT,
            UPDATE_PLAN_PROMPT,
        },
    },
};

use super::{Agent, AgentOptions, BaseAgent};

/// 规划 Agent，用于将用户的任务/需求拆解成多个子步骤。
pub struct PlannerAgent {
    base: BaseAgent,
}

impl PlannerAgent {
    /// 创建规划 Agent，并固定规划场景使用的选项。
    pub fn new(
        session_id: impl Into<String>,
        session_repository: Arc<dyn SessionRepository>,
        agent_config: AgentConfig,
        llm: Arc<dyn Llm>,
        json_parser: Arc<dyn JsonParser>,
    ) -> Self {
        Self {
            base: BaseAgent::new(
                planner_options(),
                session_id,
                session_repository,
                agent_config,
                llm,
                json_parser,
                Vec::new(),
            ),
        }
    }

    /// 根据用户传递的消息创建计划 / 规划，迭代返回对应事件。
    pub async fn create_plan(
        &mut self,
        message: Message,
        sink: &mut dyn EventSink,
    ) -> Result<EventControl> {
        // 1. 根据用户传递的消息生成创建 Plan 的提示词
        let query = render_prompt(
            CREATE_PLAN_PROMPT,
            &[
                ("{message}", &message.message),
                ("{attachments}", &message.attachments.join("\n")),
            ],
        );

        // 2. 调用 invoke 函数返回迭代事件
        let mut sink = CreatePlanSink {
            json_parser: self.base.json_parser(),
            sink,
        };
        self.base.invoke(&query, None, &mut sink).await
    }

    /// 根据传递的原始规划和已执行步骤更新后续规划。
    pub async fn update_plan(
        &mut self,
        plan: &mut Plan,
        step: &Step,
        sink: &mut dyn EventSink,
    ) -> Result<EventControl> {
        // 1. 使用 Plan 和 Step 创建更新规划提示词
        let query = render_prompt(
            UPDATE_PLAN_PROMPT,
            &[
                ("{plan}", &serde_json::to_string(plan)?),
                ("{step}", &serde_json::to_string(step)?),
            ],
        );

        // 2. 调用 invoke() 获取事件
        let mut sink = UpdatePlanSink {
            json_parser: self.base.json_parser(),
            plan,
            sink,
        };
        self.base.invoke(&query, None, &mut sink).await
    }
}

struct CreatePlanSink<'a> {
    json_parser: Arc<dyn JsonParser>,
    sink: &'a mut dyn EventSink,
}

#[async_trait]
impl EventSink for CreatePlanSink<'_> {
    async fn emit(&mut self, event: Event) -> Result<EventControl> {
        match event {
            // 3. 规划 Agent 使用 json_object，正常情况下会返回 MessageEvent
            Event::Message(message_event) => {
                // 4. 记录日志并使用 JSON 解析器解析数据
                info!(message = %message_event.message, "PlannerAgent生成消息");
                let parsed_obj = self
                    .json_parser
                    .invoke(&message_event.message, None)
                    .await?;

                // 5. 将解析对象转换成 Plan
                let plan = serde_json::from_value(parsed_obj)?;

                // 6. 返回 PlanEvent，表示规划创建成功
                self.sink
                    .emit(Event::Plan(PlanEvent {
                        plan,
                        status: PlanEventStatus::Created,
                        ..PlanEvent::default()
                    }))
                    .await
            }
            // 其他事件直接返回
            event => self.sink.emit(event).await,
        }
    }
}

struct UpdatePlanSink<'a> {
    json_parser: Arc<dyn JsonParser>,
    plan: &'a mut Plan,
    sink: &'a mut dyn EventSink,
}

#[async_trait]
impl EventSink for UpdatePlanSink<'_> {
    async fn emit(&mut self, event: Event) -> Result<EventControl> {
        match event {
            // 3. 判断规划 Agent 生成的事件是不是消息事件
            Event::Message(message_event) => {
                // 4. 记录日志并解析 JSON
                info!(message = %message_event.message, "PlannerAgent生成消息");
                let parsed_obj = self
                    .json_parser
                    .invoke(&message_event.message, None)
                    .await?;

                // 5. 将解析对象转换成 Plan
                let updated_plan: Plan = serde_json::from_value(parsed_obj)?;

                // 6. 复制更新计划中的 steps，隔离中间数据
                let new_steps = updated_plan.steps.clone();

                // 7. 查询旧计划中第一个未完成的步骤
                let first_pending_index = self.plan.steps.iter().position(|step| !step.done());

                // 8. 存在未完成步骤时，保留已完成历史并更新后续步骤
                if let Some(first_pending_index) = first_pending_index {
                    // 9. 获取历史已完成的步骤并追加新步骤
                    let mut updated_steps = self.plan.steps[..first_pending_index].to_vec();
                    updated_steps.extend(new_steps);

                    // 10. 更新 Plan
                    self.plan.steps = updated_steps;
                }

                // 11. 返回 PlanEvent，表示规划更新成功
                self.sink
                    .emit(Event::Plan(PlanEvent {
                        plan: self.plan.clone(),
                        status: PlanEventStatus::Updated,
                        ..PlanEvent::default()
                    }))
                    .await
            }
            // 其他事件直接返回
            event => self.sink.emit(event).await,
        }
    }
}

impl Agent for PlannerAgent {
    fn base(&self) -> &BaseAgent {
        &self.base
    }

    fn base_mut(&mut self) -> &mut BaseAgent {
        &mut self.base
    }
}

fn planner_options() -> AgentOptions {
    AgentOptions {
        name: "planner".to_string(),
        system_prompt: format!("{SYSTEM_PROMPT}{PLANNER_SYSTEM_PROMPT}"),
        format: Some("json_object".to_string()),
        tool_choice: Some("none".to_string()),
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
        services::agents::test_support::MemoryRepository,
    };

    impl PlannerAgent {
        pub(crate) async fn create_plan_collect(&mut self, message: Message) -> Result<Vec<Event>> {
            let mut sink =
                crate::domain::services::agents::test_support::CollectedEvents::default();
            self.create_plan(message, &mut sink).await?;
            Ok(sink.events)
        }

        pub(crate) async fn update_plan_collect(
            &mut self,
            plan: &mut Plan,
            step: &Step,
        ) -> Result<Vec<Event>> {
            let mut sink =
                crate::domain::services::agents::test_support::CollectedEvents::default();
            self.update_plan(plan, step, &mut sink).await?;
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

    fn planner(responses: Vec<Response>) -> (PlannerAgent, Requests) {
        planner_with_iterations(responses, 3)
    }

    fn planner_with_iterations(
        responses: Vec<Response>,
        max_iterations: i64,
    ) -> (PlannerAgent, Requests) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let llm = MockLlm {
            responses: Mutex::new(VecDeque::from(responses)),
            requests: Arc::clone(&requests),
        };
        let planner = PlannerAgent::new(
            "session-1",
            Arc::new(MemoryRepository::default()),
            AgentConfig {
                max_iterations,
                max_retries: 1,
                max_search_results: 10,
            },
            Arc::new(llm),
            Arc::new(MockJsonParser),
        );
        (planner, requests)
    }

    #[derive(Default)]
    struct StopSink {
        events: Vec<Event>,
    }

    #[async_trait]
    impl EventSink for StopSink {
        async fn emit(&mut self, event: Event) -> Result<EventControl> {
            self.events.push(event);
            Ok(EventControl::Stop)
        }
    }

    #[tokio::test]
    async fn created_and_updated_plan_events_propagate_stop() {
        let (mut planner, requests) = planner(vec![
            assistant_message(json!(r#"{"steps":[{"description":"first"}]}"#)),
            assistant_message(json!(r#"{"steps":[{"description":"updated"}]}"#)),
        ]);
        let mut sink = StopSink::default();

        assert_eq!(
            planner
                .create_plan(Message::default(), &mut sink)
                .await
                .unwrap(),
            EventControl::Stop
        );
        assert!(
            matches!(&sink.events[0], Event::Plan(plan) if plan.status == PlanEventStatus::Created)
        );
        let mut plan = Plan {
            steps: vec![Step::new("first")],
            ..Plan::default()
        };
        assert_eq!(
            planner
                .update_plan(&mut plan, &Step::new("done"), &mut sink)
                .await
                .unwrap(),
            EventControl::Stop
        );
        assert!(
            matches!(&sink.events[1], Event::Plan(plan) if plan.status == PlanEventStatus::Updated)
        );
        assert_eq!(plan.steps[0].description, "updated");
        assert_eq!(requests.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn planning_parse_failure_preserves_previously_delivered_error() {
        let (mut planner, _) =
            planner_with_iterations(vec![assistant_message(json!("broken json"))], 0);
        let mut sink = crate::domain::services::agents::test_support::CollectedEvents::default();

        assert!(planner
            .create_plan(Message::default(), &mut sink)
            .await
            .is_err());

        assert_eq!(sink.events.len(), 1);
        assert!(
            matches!(&sink.events[0], Event::Error(error) if error.error.contains("最大迭代次数"))
        );
    }

    #[test]
    fn planner_options_match_python_defaults() {
        let options = planner_options();

        assert_eq!(options.name, "planner");
        assert_eq!(options.format.as_deref(), Some("json_object"));
        assert_eq!(options.tool_choice.as_deref(), Some("none"));
        assert!(options.system_prompt.contains(SYSTEM_PROMPT));
        assert!(options.system_prompt.contains(PLANNER_SYSTEM_PROMPT));
    }

    #[tokio::test]
    async fn create_plan_converts_message_event_into_created_plan_event() {
        let (mut planner, requests) = planner(vec![assistant_message(json!(
            r#"{
                "message":"计划已生成",
                "language":"zh",
                "steps":[{"id":"1","description":"执行发布检查"}],
                "goal":"完成应用发布",
                "title":"发布应用"
            }"#
        ))]);

        let events = planner
            .create_plan_collect(Message {
                message: "发布应用".to_string(),
                attachments: vec!["brief.md".to_string(), "budget.xlsx".to_string()],
            })
            .await
            .unwrap();

        assert_eq!(events.len(), 1);
        let Event::Plan(event) = &events[0] else {
            panic!("事件必须是创建规划事件");
        };
        assert_eq!(event.status, PlanEventStatus::Created);
        assert_eq!(event.plan.title, "发布应用");
        assert_eq!(event.plan.steps.len(), 1);
        assert_eq!(event.plan.steps[0].description, "执行发布检查");

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
        assert_eq!(request.tool_choice.as_deref(), Some("none"));
        let query = request
            .messages
            .last()
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .unwrap();
        assert!(query.contains("发布应用"));
        assert!(query.contains("brief.md\nbudget.xlsx"));
        assert!(!query.contains("{message}"));
        assert!(!query.contains("{attachments}"));
    }

    #[tokio::test]
    async fn update_plan_keeps_completed_history_and_replaces_pending_steps() {
        let (mut planner, _) = planner(vec![assistant_message(json!(
            r#"{
                "steps":[
                    {"id":"2","description":"重新执行部署"},
                    {"id":"3","description":"执行验收"}
                ]
            }"#
        ))]);
        let completed = Step {
            id: "1".to_string(),
            description: "完成构建".to_string(),
            status: crate::domain::models::ExecutionStatus::Completed,
            ..Step::default()
        };
        let running = Step {
            id: "2".to_string(),
            description: "执行部署".to_string(),
            ..Step::default()
        };
        let pending = Step {
            id: "3".to_string(),
            description: "旧的验收步骤".to_string(),
            ..Step::default()
        };
        let mut plan = Plan {
            goal: "发布应用".to_string(),
            steps: vec![completed.clone(), running, pending],
            ..Plan::default()
        };
        let executed_step = Step {
            id: "1".to_string(),
            description: "完成构建".to_string(),
            status: crate::domain::models::ExecutionStatus::Completed,
            success: true,
            ..Step::default()
        };

        let events = planner
            .update_plan_collect(&mut plan, &executed_step)
            .await
            .unwrap();

        assert_eq!(events.len(), 1);
        let Event::Plan(event) = &events[0] else {
            panic!("事件必须是更新规划事件");
        };
        assert_eq!(event.status, PlanEventStatus::Updated);
        assert_eq!(
            plan.steps
                .iter()
                .map(|step| step.description.as_str())
                .collect::<Vec<_>>(),
            vec!["完成构建", "重新执行部署", "执行验收"]
        );
        assert_eq!(plan.steps[0], completed);
        assert_eq!(event.plan, plan);
        assert_eq!(plan.goal, "发布应用");
    }

    #[tokio::test]
    async fn create_plan_passes_through_non_message_events() {
        let (mut planner, _) = planner(vec![LlmMessage::from_iter([(
            "role".to_string(),
            json!("tool"),
        )])]);

        let events = planner
            .create_plan_collect(Message::default())
            .await
            .unwrap();

        assert_eq!(events.len(), 1);
        let Event::Error(event) = &events[0] else {
            panic!("事件必须原样透传");
        };
        assert_eq!(event.error, "Agent未能生成有效回复内容");
    }

    #[tokio::test]
    async fn planner_keeps_template_like_text_in_messages_and_serialized_plans() {
        let (mut planner, requests) = planner(vec![
            assistant_message(json!(r#"{"steps":[]}"#)),
            assistant_message(json!(r#"{"steps":[]}"#)),
        ]);
        let user_message = "请解释 {attachments} 与 {step}，保留 {{原文}}";
        planner
            .create_plan_collect(Message {
                message: user_message.into(),
                attachments: vec!["/home/ubuntu/{message}.md".into()],
            })
            .await
            .unwrap();
        let mut plan = Plan {
            goal: "解释 {step}".into(),
            steps: vec![Step::new("分析 {plan}")],
            ..Plan::default()
        };
        let serialized_plan = serde_json::to_string(&plan).unwrap();
        let step = Step::new("处理 {message}");
        let serialized_step = serde_json::to_string(&step).unwrap();
        planner.update_plan_collect(&mut plan, &step).await.unwrap();

        let requests = requests.lock().unwrap();
        let create_query = requests[0].messages.last().unwrap()["content"]
            .as_str()
            .unwrap();
        assert!(create_query.contains(user_message));
        assert!(create_query.contains("/home/ubuntu/{message}.md"));
        let update_query = requests[1].messages.last().unwrap()["content"]
            .as_str()
            .unwrap();
        assert!(update_query.contains(&serialized_plan));
        assert!(update_query.contains(&serialized_step));
    }
}
