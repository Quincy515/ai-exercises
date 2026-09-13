use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use crate::domain::{
    external::{
        Browser, FileStorage, JsonParser, Llm, Sandbox, SearchEngine, SharedTask, Task, TaskRunner,
    },
    models::{A2aConfig, AgentConfig, ErrorEvent, Event, McpConfig, SessionStatus},
    repositories::{FileRepository, SessionRepository},
    services::{
        flows::PlannerReActFlow,
        tools::{A2ATool, McpTool},
    },
};

/// 基于 Agent 智能体的任务运行器。
pub struct AgentTaskRunner {
    /// 会话 id
    session_id: String,
    /// 会话仓库
    session_repository: Arc<dyn SessionRepository>,
    /// 沙箱
    sandbox: Arc<dyn Sandbox>,
    /// 文件存储桶，供后续附件同步使用
    _file_storage: Arc<dyn FileStorage>,
    /// 文件数据仓库，供后续附件同步使用
    _file_repository: Arc<dyn FileRepository>,
    /// 规划与执行流；异步锁协调任务调用和资源销毁。
    flow: Mutex<PlannerReActFlow>,
}

impl AgentTaskRunner {
    /// 构造函数，完成 Agent 任务运行器的创建。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        llm: Arc<dyn Llm>,                              // 大语言模型
        agent_config: AgentConfig,                      // 智能体配置
        mcp_config: McpConfig,                          // MCP 配置
        a2a_config: A2aConfig,                          // A2A 配置
        session_id: impl Into<String>,                  // 会话 id
        session_repository: Arc<dyn SessionRepository>, // 会话仓库
        file_storage: Arc<dyn FileStorage>,             // 文件存储桶
        file_repository: Arc<dyn FileRepository>,       // 文件数据仓库
        json_parser: Arc<dyn JsonParser>,               // JSON 解析器
        browser: Box<dyn Browser>,                      // 浏览器
        search_engine: Box<dyn SearchEngine>,           // 搜索引擎
        sandbox: Arc<dyn Sandbox>,                      // 沙箱
    ) -> Self {
        let session_id = session_id.into();
        // 浏览器和远程工具随流持有，运行器通过流初始化、清理同一组工具实例。
        let flow = PlannerReActFlow::new(
            llm,
            agent_config,
            session_id.clone(),
            session_repository.clone(),
            json_parser,
            browser,
            sandbox.clone(),
            search_engine,
            McpTool::with_config(Some(mcp_config)),
            A2ATool::with_config(Some(a2a_config)),
        );

        Self {
            session_id,
            session_repository,
            sandbox,
            _file_storage: file_storage,
            _file_repository: file_repository,
            flow: Mutex::new(flow),
        }
    }

    /// 往指定任务的消息队列中添加事件。
    async fn put_and_add_event(&self, task: &dyn Task, mut event: Event) -> Result<()> {
        // 1.往任务的输出消息队列中新增事件。
        // 队列协议接收 JSON 值，Redis 适配器负责将其编码为字符串。
        let event_id = task
            .output_stream()
            .put(serde_json::to_value(&event)?)
            .await?;
        event.set_id(event_id);

        // 2.将事件添加到对应的会话中。
        self.session_repository
            .add_event(&self.session_id, event)
            .await
    }

    /// 从任务的输入流中获取事件信息。
    async fn pop_event(task: &dyn Task) -> Result<Option<Event>> {
        // 1.从任务 task 中读取数据。
        let Some((event_id, payload)) = task.input_stream().pop().await? else {
            warn!("AgentTaskRunner接收到空消息");
            return Ok(None);
        };

        // 2.使用 Serde 和 type 字段将队列数据转换成具体事件。
        let mut event: Event = serde_json::from_value(payload)?;
        event.set_id(event_id);
        Ok(Some(event))
    }
}

#[async_trait]
impl TaskRunner for AgentTaskRunner {
    /// 根据传递的任务处理 Agent 消息队列并运行 Agent 流。
    async fn invoke(&self, task: SharedTask) -> Result<()> {
        let result: Result<()> = async {
            // 同一次运行和销毁共用一把锁，避免资源在初始化或使用中被清理。
            let mut flow = self.flow.lock().await;

            // 1.确保沙箱、MCP、A2A 均初始化完成。
            info!("AgentTaskRunner任务处理开始");
            self.sandbox.ensure_sandbox().await?;
            flow.initialize_tools().await?;

            // 2.循环读取任务中的输入消息队列。
            while !task.input_stream().is_empty().await? {
                // 3.从输入流中获取数据。
                let Some(_event) = Self::pop_event(task.as_ref()).await? else {
                    continue;
                };
                let _message = String::new();

                // todo:后续的逻辑待实现（附件同步、消息转换、调用流和事件处理）。
            }

            Ok(())
        }
        .await;

        if let Err(error) = result {
            // 记录日志并往任务队列/消息队列中写入异常事件并更新会话状态。
            error!(error = %error, "AgentTaskRunner运行出错");
            self.put_and_add_event(
                task.as_ref(),
                Event::Error(ErrorEvent {
                    error: format!("AgentTaskRunner出错: {error}"),
                    ..ErrorEvent::default()
                }),
            )
            .await?;
            self.session_repository
                .update_status(&self.session_id, SessionStatus::Completed)
                .await?;
        }

        Ok(())
    }

    /// 销毁任务运行器并释放资源。
    async fn destroy(&self) -> Result<()> {
        let mut flow = self.flow.lock().await;
        info!("开始清除销毁AgentTaskRunner资源");

        // 1.清除沙箱。
        info!("销毁AgentTaskRunner中的沙箱环境");
        self.sandbox.destroy().await?;

        // 2.清除 MCP 工具；3.清除 A2A 工具。由流按此顺序访问原工具实例。
        flow.cleanup_tools().await?;
        Ok(())
    }

    /// 任务结束时执行的回调函数。
    async fn on_done(&self, _task: SharedTask) -> Result<()> {
        info!("AgentTaskRunner任务执行结束");
        Ok(())
    }
}

#[cfg(test)]
#[path = "agent_task_runner_tests.rs"]
mod tests;
