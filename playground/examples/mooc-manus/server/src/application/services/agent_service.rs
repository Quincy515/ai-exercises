use std::{sync::Arc, time::Duration};

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use futures::{stream, stream::BoxStream, StreamExt};

use crate::application::shutdown::ShutdownSignal;

use crate::domain::{
    external::{
        FileStorage, JsonParser, Llm, SandboxFactory, SearchEngine, SharedTask, Task, TaskFactory,
    },
    models::{
        A2aConfig, AgentConfig, ErrorEvent, Event, File, McpConfig, MessageEvent, MessageRole,
        Session, SessionStatus,
    },
    repositories::{FileRepository, SessionRepository},
    services::agent_task_runner::AgentTaskRunner,
};

/// Manus 智能体服务，编排会话、沙箱与后台任务。
pub struct AgentService {
    shutdown: ShutdownSignal,
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
    /// 关闭 Agent 服务，清除所有会话任务资源。
    /// Task 的静态销毁入口由应用装配层指定，退出时复用已注册的任务。
    pub async fn shutdown<T: Task>(shutdown: &ShutdownSignal) -> Result<()> {
        tracing::info!("正在清除所有会话任务资源并释放");
        shutdown.notify();
        let _preparations = shutdown.finish_preparations().await;
        T::destroy().await?;
        tracing::info!("所有会话任务资源清除成功");
        Ok(())
    }

    /// 在独立后台任务中安全更新未读数，任务拥有连接池仓库句柄。
    /// HTTP Stream 被丢弃时，该任务继续执行自己的短数据库操作。
    async fn safe_update_unread_count(
        session_repository: Arc<dyn SessionRepository>,
        session_id: String,
    ) {
        if let Err(error) = session_repository
            .update_unread_message_count(&session_id, 0)
            .await
        {
            tracing::warn!(session_id, error = %error, "后台更新未读消息计数失败");
        }
    }

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
        shutdown: ShutdownSignal,
    ) -> Self {
        tracing::info!("AgentService 初始化成功");
        Self {
            shutdown,
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

    /// 根据传递的会话 id 停止指定会话。
    pub async fn stop_session(&self, session_id: &str) -> Result<()> {
        // 1.查找会话是否存在。
        let session = self.session_repository.get_by_id(session_id).await?;
        let Some(session) = session else {
            tracing::error!(session_id, "尝试停止不存在的会话");
            return Err(anyhow!("任务会话不存在, 请核实后重试"));
        };

        // 2.根据会话获取任务信息，有任务时发出取消请求。
        if let Some(task) = self.get_task(&session)? {
            task.cancel();
        }

        // 3.更新会话任务状态；任务已经结束或不在注册表中时同样标记完成。
        self.session_repository
            .update_status(session_id, SessionStatus::Completed)
            .await
    }

    /// 根据传递的信息调用 Agent 服务发起对话请求，逐个返回任务输出事件。
    pub fn chat(
        self,
        session_id: String,
        message: Option<String>,
        attachments: Option<Vec<String>>,
        latest_event_id: Option<String>,
        timestamp: Option<DateTime<Utc>>,
    ) -> BoxStream<'static, Event> {
        let state = ChatStream {
            service: self,
            session_id,
            request: Some(ChatInput {
                message,
                attachments,
                timestamp,
            }),
            task: None,
            latest_event_id,
            finished: false,
            started: false,
            cleanup_scheduled: false,
        };
        // Stream 被读取时才启动本轮；丢弃订阅后后台任务继续运行。
        stream::unfold(state, |mut state| async move {
            if state.finished {
                return None;
            }
            state.started = true;
            match state.next_event().await {
                Ok(Some(event)) => {
                    // 15.返回事件，Done、Error、Wait 均结束本轮订阅。
                    state.finished = matches!(event, Event::Done(_) | Event::Error(_) | Event::Wait(_));
                    if state.finished { state.finish(); }
                    Some((event, state))
                }
                Ok(None) => {
                    state.finish();
                    None
                }
                Err(error) => {
                    // 17.记录日志，持久化并返回错误事件。
                    tracing::error!(session_id = %state.session_id, error = %error, "任务会话对话出错");
                    let event = Event::Error(ErrorEvent {
                        error: error.to_string(),
                        ..ErrorEvent::default()
                    });
                    if let Err(save_error) = state.service.session_repository
                        .add_event(&state.session_id, event.clone()).await
                    {
                        tracing::warn!(session_id = %state.session_id, error = %save_error, "保存会话错误事件失败");
                    }
                    state.finished = true;
                    state.finish();
                    Some((event, state))
                }
            }
        }).boxed()
    }

    async fn prepare_chat(
        &self,
        session_id: &str,
        request: ChatInput,
    ) -> Result<Option<SharedTask>> {
        // 与关闭时的任务快照同步，防止快照之后才注册新任务。
        let _preparing = self
            .shutdown
            .begin_preparation()
            .await
            .context("Agent服务正在关闭")?;
        // 1.检查会话是否存在。
        let mut session = self
            .session_repository
            .get_by_id(session_id)
            .await?
            .ok_or_else(|| anyhow!("任务会话不存在, 请核实后重试"))?;
        // 2.获取对应会话任务。
        let mut task = self.get_task(&session)?;
        // 3.判断是否传递了非空 message。
        if let Some(message) = request.message.filter(|message| !message.is_empty()) {
            // 4.会话不在运行中，或进程重启后任务实例已丢失时，创建新任务。
            if session.status != SessionStatus::Running || task.is_none() {
                // 5.创建并保存新 Task。
                task = Some(self.create_task(&mut session).await?);
            }
            let current_task = task
                .as_ref()
                .context("会话运行中的任务实例不存在，请核实后重试")?;
            // 6.更新最后一条消息；省略时间戳时使用服务端当前时间。
            self.session_repository
                .update_latest_message(
                    session_id,
                    &message,
                    request.timestamp.unwrap_or_else(Utc::now),
                )
                .await?;
            // 7.创建人类消息事件，附件传入的是文件 id。
            let mut event = Event::Message(MessageEvent {
                role: MessageRole::User,
                message: message.clone(),
                attachments: request
                    .attachments
                    .unwrap_or_default()
                    .into_iter()
                    .map(|id| File {
                        id,
                        ..File::default()
                    })
                    .collect(),
                ..MessageEvent::default()
            });
            // 8.将事件写入输入流，使用队列返回的 id 保存会话事件。
            let event_id = current_task
                .input_stream()
                .put(serde_json::to_value(&event)?)
                .await?;
            event.set_id(event_id);
            self.session_repository.add_event(session_id, event).await?;
            // 9.启动后台任务；已有任务正在执行时由 Task 自身保证重复调用安全。
            current_task.invoke().await?;
            tracing::info!(session_id, message = %message.chars().take(50).collect::<String>(), "往会话输入消息队列写入消息");
        }
        // 10.记录本轮订阅关联的任务。
        tracing::info!(session_id, task_id = ?task.as_ref().map(|task| task.id()), "会话开始读取任务事件");
        Ok(task)
    }
}

/// 保存惰性流首次执行所需的业务参数。
struct ChatInput {
    message: Option<String>,
    attachments: Option<Vec<String>>,
    timestamp: Option<DateTime<Utc>>,
}

/// 每个 HTTP 订阅独立保存读取游标，后台 Task 由任务注册表持有。
struct ChatStream {
    service: AgentService,
    session_id: String,
    request: Option<ChatInput>,
    task: Option<SharedTask>,
    latest_event_id: Option<String>,
    finished: bool,
    started: bool,
    cleanup_scheduled: bool,
}

impl ChatStream {
    async fn next_event(&mut self) -> Result<Option<Event>> {
        if let Some(request) = self.request.take() {
            self.task = self.service.prepare_chat(&self.session_id, request).await?;
        }
        let Some(task) = &self.task else {
            return Ok(None);
        };
        // 11.逐条读取输出；任务已完成时也要排空已经写入队列的事件。
        loop {
            let was_done = task.done();
            // 12.输入和输出共享 Redis 连接，非阻塞读取避免 BLOCK 0 阻塞后台写入。
            let item = task
                .output_stream()
                .get(self.latest_event_id.as_deref(), None)
                .await?;
            if let Some((event_id, payload)) = item {
                self.latest_event_id = Some(event_id.clone());
                // 13.按 type 解码领域事件，并使用队列返回的实际事件 id。
                let mut event: Event = serde_json::from_value(payload)?;
                event.set_id(event_id);
                // 14.当前订阅正在传递事件，重置未读消息数。
                self.service
                    .session_repository
                    .update_unread_message_count(&self.session_id, 0)
                    .await?;
                return Ok(Some(event));
            }
            if was_done {
                return Ok(None);
            }
            // 空读期间任务可能刚结束，下一次读取会再次检查其输出。
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    fn finish(&mut self) {
        if self.cleanup_scheduled {
            return;
        }
        self.cleanup_scheduled = true;
        // 16.本轮订阅结束。
        tracing::info!(session_id = %self.session_id, "会话本轮运行结束");
        // 18.正常、异常和客户端断连出口均启动独立任务清零，失败只记日志。
        // Rust Drop 无法 await；新任务独立持有仓库，使用连接池获取自己的连接。
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => {
                runtime.spawn(AgentService::safe_update_unread_count(
                    self.service.session_repository.clone(),
                    self.session_id.clone(),
                ));
            }
            Err(error) => tracing::warn!(session_id = %self.session_id, error = %error,
                "无法创建后台任务更新未读消息计数"),
        }
    }
}

impl Drop for ChatStream {
    fn drop(&mut self) {
        // 已开始订阅的 Stream 在等待中被丢弃时补上 finally；后台 Agent Task 继续运行。
        if self.started {
            self.finish();
        }
    }
}
