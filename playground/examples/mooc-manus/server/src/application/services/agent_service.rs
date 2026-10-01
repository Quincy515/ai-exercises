use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use futures::{stream, stream::BoxStream, StreamExt};

use crate::domain::{
    external::{
        FileStorage, JsonParser, Llm, SandboxFactory, SearchEngine, SharedTask, TaskFactory,
    },
    models::{A2aConfig, AgentConfig, ErrorEvent, Event, McpConfig, Session, SessionStatus},
    repositories::{FileRepository, SessionRepository},
    services::agent_task_runner::AgentTaskRunner,
};

/// Manus 智能体服务，编排会话、沙箱与后台任务。
pub struct AgentService {
    session_repository: Arc<dyn SessionRepository>,
    llm: Arc<dyn Llm>,
    agent_config: AgentConfig,
    mcp_config: McpConfig,
    a2a_config: A2aConfig,
    sandbox_factory: Arc<dyn SandboxFactory>,
    task_factory: Arc<dyn TaskFactory>,
    json_parser: Arc<dyn JsonParser>,
    search_engine: Arc<dyn SearchEngine>,
    file_storage: Arc<dyn FileStorage>,
    file_repository: Arc<dyn FileRepository>,
}

impl AgentService {
    /// 构造函数，完成 Agent 服务初始化。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session_repository: Arc<dyn SessionRepository>,
        llm: Arc<dyn Llm>,
        agent_config: AgentConfig,
        mcp_config: McpConfig,
        a2a_config: A2aConfig,
        sandbox_factory: Arc<dyn SandboxFactory>,
        task_factory: Arc<dyn TaskFactory>,
        json_parser: Arc<dyn JsonParser>,
        search_engine: Arc<dyn SearchEngine>,
        file_storage: Arc<dyn FileStorage>,
        file_repository: Arc<dyn FileRepository>,
    ) -> Self {
        tracing::info!("AgentService 初始化成功");
        Self {
            session_repository,
            llm,
            agent_config,
            mcp_config,
            a2a_config,
            sandbox_factory,
            task_factory,
            json_parser,
            search_engine,
            file_storage,
            file_repository,
        }
    }

    /// 根据传递的任务会话获取任务实例。
    fn get_task(&self, session: &Session) -> Result<Option<SharedTask>> {
        // 1.从会话中取出任务 id。
        let Some(task_id) = session.task_id.as_deref().filter(|id| !id.is_empty()) else {
            return Ok(None);
        };
        // 2.调用任务工厂获取对应的任务实例。
        self.task_factory.get(task_id)
    }

    /// 根据传递的会话创建一个新任务。
    async fn create_task(&self, session: &mut Session) -> Result<SharedTask> {
        // 1.获取沙箱实例。
        let sandbox = match session.sandbox_id.as_deref().filter(|id| !id.is_empty()) {
            Some(sandbox_id) => self.sandbox_factory.get(sandbox_id).await?,
            None => None,
        };

        // 2.判断是否能获取到沙箱，如果没有则创建。
        let sandbox = match sandbox {
            Some(sandbox) => sandbox,
            None => {
                // 3.沙箱不存在则创建一个新的，有可能旧沙箱已经被释放。
                let sandbox = self.sandbox_factory.create().await?;
                session.sandbox_id = Some(sandbox.id().to_owned());
                self.session_repository.save(session.clone()).await?;
                sandbox
            }
        };

        // 4.从沙箱中获取浏览器实例；错误向外传播为会话错误事件。
        let browser = sandbox
            .get_browser()
            .await
            .with_context(|| format!("获取沙箱[{}]中的浏览器实例失败", sandbox.id()))?;

        // 5.创建 AgentTaskRunner。
        let task_runner = Arc::new(AgentTaskRunner::new(
            self.llm.clone(),
            self.agent_config.clone(),
            self.mcp_config.clone(),
            self.a2a_config.clone(),
            session.id.clone(),
            self.session_repository.clone(),
            self.file_storage.clone(),
            self.file_repository.clone(),
            self.json_parser.clone(),
            browser,
            Box::new(self.search_engine.clone()),
            Arc::from(sandbox),
        ));

        // 6.创建 Task 并更新会话信息，task_id 保存任务的字符串标识。
        let task = self
            .task_factory
            .create(task_runner)
            .await
            .with_context(|| format!("会话[{}]创建任务失败", session.id))?;
        session.task_id = Some(task.id().to_owned());
        self.session_repository.save(session.clone()).await?;
        Ok(task)
    }

    /// 根据传递的信息调用 Agent 服务发起对话请求。
    /// 本课先准备任务；附件、事件游标和时间戳的处理由后续课时接入。
    pub fn chat(
        self,
        session_id: String,
        message: Option<String>,
        _attachments: Option<Vec<String>>,
        _latest_event_id: Option<String>,
        _timestamp: Option<i64>,
    ) -> BoxStream<'static, Event> {
        // Stream 在响应体被读取时才执行，成功分支本课暂不产生业务事件。
        stream::once(async move {
            let result = self.prepare_chat(&session_id, message.as_deref()).await;
            if let Err(error) = result {
                // 记录日志并返回错误事件。
                tracing::error!(session_id, error = %error, "任务会话对话出错");
                let event = Event::Error(ErrorEvent {
                    error: error.to_string(),
                    ..ErrorEvent::default()
                });
                if let Err(save_error) = self
                    .session_repository
                    .add_event(&session_id, event.clone())
                    .await
                {
                    // 会话缺失时也能保留原始错误，追加失败不覆盖本次错误事件。
                    tracing::error!(session_id, error = %save_error, "保存会话错误事件失败");
                }
                Some(event)
            } else {
                None
            }
        })
        .filter_map(futures::future::ready)
        .boxed()
    }

    async fn prepare_chat(&self, session_id: &str, message: Option<&str>) -> Result<()> {
        // 1.检查会话是否存在。
        let mut session = self
            .session_repository
            .get_by_id(session_id)
            .await?
            .ok_or_else(|| anyhow!("任务会话不存在, 请核实后重试"))?;

        // 2.获取对应会话任务。
        let mut task = self.get_task(&session)?;

        // 3.判断是否传递了非空 message。
        if message.is_some_and(|message| !message.is_empty()) {
            // 4.会话不在运行中时，准备一个新任务。
            if session.status != SessionStatus::Running {
                // 5.创建新 Task，启动和消息处理在后续课时完成。
                task = Some(self.create_task(&mut session).await?);
                // TODO: 后续逻辑待实现。
            }
        }
        let _ = task;
        Ok(())
    }
}
