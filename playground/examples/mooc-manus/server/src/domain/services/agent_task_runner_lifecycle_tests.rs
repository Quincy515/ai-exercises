//! 通过真实 Runner/Flow 验证任务结束后的资源收尾；模型与远程工具均使用内存替身。

use super::*;
use crate::domain::services::{
    agents::{PlannerAgent, ReActAgent},
    flows::PlannerReActFlow,
    tools::{BaseTool, MessageTool, ToolArguments, ToolDefinition},
};

struct LifecycleTool {
    name: &'static str,
    calls: Arc<Mutex<Vec<String>>>,
    fail_cleanup: bool,
}

#[async_trait]
impl BaseTool for LifecycleTool {
    fn name(&self) -> &str {
        self.name
    }

    fn tool_definitions(&self) -> &[ToolDefinition] {
        &[]
    }

    async fn initialize(&mut self) -> Result<()> {
        tracing::debug!(tool = self.name, "初始化生命周期测试工具");
        self.calls
            .lock()
            .unwrap()
            .push(format!("{}:initialize", self.name));
        Ok(())
    }

    async fn call_tool(
        &self,
        _tool_name: &str,
        _kwargs: ToolArguments,
    ) -> Result<ToolResult<Value>> {
        panic!("生命周期测试只调用 MessageTool")
    }

    async fn cleanup(&mut self) -> Result<()> {
        tracing::debug!(tool = self.name, "清理生命周期测试工具");
        self.calls
            .lock()
            .unwrap()
            .push(format!("{}:cleanup", self.name));
        if self.fail_cleanup {
            bail!("模拟{}清理失败", self.name);
        }
        Ok(())
    }
}

struct LifecycleFixture {
    runner: AgentTaskRunner,
    repository: Arc<MemoryRepository>,
    sandbox: Arc<LifecycleSandbox>,
    calls: Arc<Mutex<Vec<String>>>,
}

fn lifecycle_fixture(llm: Arc<dyn Llm>, fail_mcp: bool, fail_a2a: bool) -> LifecycleFixture {
    let (mut runner, repository, sandbox) = fixture();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let config = AgentConfig {
        max_retries: 1,
        max_iterations: 2,
        ..AgentConfig::default()
    };
    let parser: Arc<dyn JsonParser> = Arc::new(TestJsonParser);
    let planner = PlannerAgent::new(
        SESSION_ID,
        repository.clone(),
        config.clone(),
        llm.clone(),
        parser.clone(),
    );
    // 注册顺序刻意与清理顺序相反，验证按照工具名称释放原实例。
    let tools: Vec<Box<dyn BaseTool>> = vec![
        Box::new(MessageTool::new()),
        Box::new(LifecycleTool {
            name: "a2a",
            calls: calls.clone(),
            fail_cleanup: fail_a2a,
        }),
        Box::new(LifecycleTool {
            name: "mcp",
            calls: calls.clone(),
            fail_cleanup: fail_mcp,
        }),
    ];
    let react = ReActAgent::new(SESSION_ID, repository.clone(), config, llm, parser, tools);
    runner.flow = tokio::sync::Mutex::new(PlannerReActFlow::from_agents(
        SESSION_ID,
        repository.clone(),
        planner,
        react,
    ));
    LifecycleFixture {
        runner,
        repository,
        sandbox,
        calls,
    }
}

fn assert_initialized_and_cleaned(calls: &Mutex<Vec<String>>) {
    assert_eq!(
        *calls.lock().unwrap(),
        [
            "a2a:initialize",
            "mcp:initialize",
            "mcp:cleanup",
            "a2a:cleanup"
        ]
    );
}

#[tokio::test]
async fn invoke_cleans_remote_tools_before_returning_from_completion_error_and_wait() {
    for outcome in ["completed", "error", "wait"] {
        let llm: Arc<dyn Llm> = match outcome {
            "wait" => Arc::new(WaitingLlm {
                requests: AtomicUsize::new(0),
            }),
            "error" => {
                // 首次模型生成计划，第二次模型失败，覆盖 Runner 的 Err 捕获路径。
                let release = Arc::new(tokio::sync::Notify::new());
                release.notify_one();
                Arc::new(BlockingPlanLlm {
                    calls: AtomicUsize::new(0),
                    entered: Arc::new(tokio::sync::Notify::new()),
                    release,
                })
            }
            _ => Arc::new(PlanningLlm::default()),
        };
        let fixture = lifecycle_fixture(llm, false, false);
        let task = Arc::new(MemoryTask::default());
        task.input
            .push("input-1", serde_json::to_value(message()).unwrap());

        timeout(Duration::from_secs(2), fixture.runner.invoke(task.clone()))
            .await
            .unwrap()
            .unwrap();
        let before = fixture.repository.session(SESSION_ID).unwrap();
        let output_before = task.output.entries();
        assert_eq!(before.events.len(), output_before.len(), "{outcome}");
        match outcome {
            "completed" => {
                assert_eq!(before.status, SessionStatus::Completed);
                assert!(matches!(before.events.last(), Some(Event::Done(_))));
            }
            "error" => {
                assert_eq!(before.status, SessionStatus::Completed);
                assert!(matches!(before.events.last(), Some(Event::Error(_))));
                assert_eq!(
                    output_before
                        .iter()
                        .map(|(_, event)| event["type"].as_str().unwrap())
                        .collect::<Vec<_>>(),
                    ["title", "message", "plan", "step", "error"]
                );
            }
            "wait" => {
                assert_eq!(before.status, SessionStatus::Waiting);
                assert!(matches!(before.events.last(), Some(Event::Wait(_))));
            }
            _ => unreachable!(),
        }

        // invoke 的 finally 等价出口已经清理完成，on_done 只记录完成日志。
        assert_initialized_and_cleaned(&fixture.calls);
        timeout(Duration::from_secs(2), fixture.runner.on_done(task.clone()))
            .await
            .unwrap()
            .unwrap();
        assert_initialized_and_cleaned(&fixture.calls);
        assert_eq!(*fixture.sandbox.calls.lock().unwrap(), ["ensure"]);
        assert_eq!(fixture.repository.session(SESSION_ID).unwrap(), before);
        assert_eq!(task.output.entries(), output_before);
    }
}

#[tokio::test]
async fn cancellation_releases_flow_lock_keeps_published_prefix_and_saves_done() {
    let entered = Arc::new(tokio::sync::Notify::new());
    let llm = Arc::new(BlockingPlanLlm {
        calls: AtomicUsize::new(0),
        entered: entered.clone(),
        release: Arc::new(tokio::sync::Notify::new()),
    });
    let fixture = lifecycle_fixture(llm.clone(), false, false);
    let runner = Arc::new(fixture.runner);
    let task = Arc::new(MemoryTask::default());
    task.input
        .push("input-1", serde_json::to_value(message()).unwrap());
    let running = tokio::spawn({
        let runner = runner.clone();
        let task = task.clone();
        async move { runner.invoke(task).await }
    });
    let reached = timeout(Duration::from_secs(2), entered.notified()).await;
    let prefix = fixture.repository.session(SESSION_ID).unwrap().events;
    let output_prefix = task.output.entries();
    let still_running = !running.is_finished();

    // 模拟 Tokio supervisor 的时序：abort 后等待 future 丢弃，再执行回调。
    // 即使 gate 未及时到达，也先终止执行并清理资源，再报告断言失败。
    running.abort();
    let joined = timeout(Duration::from_secs(2), running).await;
    let cancelled = timeout(Duration::from_secs(2), runner.on_cancel(task.clone())).await;
    let cleaned = timeout(Duration::from_secs(2), runner.on_done(task.clone())).await;

    reached.unwrap();
    let error = joined.unwrap().unwrap_err();
    assert!(error.is_cancelled());
    cancelled.unwrap().unwrap();
    cleaned.unwrap().unwrap();
    assert!(still_running);
    assert_eq!(
        output_prefix
            .iter()
            .map(|(_, event)| event["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["title", "message", "plan", "step"]
    );

    let session = fixture.repository.session(SESSION_ID).unwrap();
    assert_eq!(session.status, SessionStatus::Completed);
    assert_eq!(session.events[..prefix.len()], prefix);
    assert_eq!(session.events.len(), prefix.len() + 1);
    assert!(matches!(session.events.last(), Some(Event::Done(_))));
    let output = task.output.entries();
    assert_eq!(output[..output_prefix.len()], output_prefix);
    assert_eq!(output.last().unwrap().1["type"], "done");
    assert_eq!(output.len(), session.events.len());
    let Some(Event::Done(done)) = session.events.last() else {
        unreachable!()
    };
    assert_eq!(done.base.id, output.last().unwrap().0);
    assert_eq!(llm.calls.load(Ordering::SeqCst), 2);
    assert_initialized_and_cleaned(&fixture.calls);
    assert_eq!(*fixture.sandbox.calls.lock().unwrap(), ["ensure"]);
}

#[tokio::test]
async fn invoke_logs_cleanup_errors_continues_and_on_done_preserves_the_session() {
    for (fail_mcp, fail_a2a) in [(true, false), (false, true), (true, true)] {
        let fixture = lifecycle_fixture(Arc::new(PlanningLlm::default()), fail_mcp, fail_a2a);
        let task = Arc::new(MemoryTask::default());
        fixture.runner.invoke(task.clone()).await.unwrap();
        let before = fixture.repository.session(SESSION_ID).unwrap();

        assert_initialized_and_cleaned(&fixture.calls);
        timeout(Duration::from_secs(2), fixture.runner.on_done(task))
            .await
            .unwrap()
            .unwrap();

        assert_initialized_and_cleaned(&fixture.calls);
        assert_eq!(*fixture.sandbox.calls.lock().unwrap(), ["ensure"]);
        assert_eq!(fixture.repository.session(SESSION_ID).unwrap(), before);
    }
}

#[tokio::test]
async fn destroy_stops_on_sandbox_error_and_logs_remote_cleanup_errors() {
    for (fail_sandbox, fail_mcp, fail_a2a, expected_error) in [
        (false, false, false, None),
        (true, true, true, Some("模拟沙箱销毁失败")),
        (false, true, true, None),
        (false, false, true, None),
    ] {
        let fixture = lifecycle_fixture(Arc::new(PlanningLlm::default()), fail_mcp, fail_a2a);
        fixture
            .runner
            .invoke(Arc::new(MemoryTask::default()))
            .await
            .unwrap();
        fixture.calls.lock().unwrap().clear();
        fixture
            .sandbox
            .fail_destroy
            .store(fail_sandbox, Ordering::SeqCst);

        let result = timeout(Duration::from_secs(2), fixture.runner.destroy())
            .await
            .unwrap();

        match expected_error {
            Some(expected) => assert_eq!(result.unwrap_err().to_string(), expected),
            None => result.unwrap(),
        }
        let expected: Vec<&str> = if fail_sandbox {
            vec![]
        } else {
            vec!["mcp:cleanup", "a2a:cleanup"]
        };
        assert_eq!(*fixture.calls.lock().unwrap(), expected);
        assert_eq!(
            *fixture.sandbox.calls.lock().unwrap(),
            ["ensure", "destroy"]
        );
    }
}
