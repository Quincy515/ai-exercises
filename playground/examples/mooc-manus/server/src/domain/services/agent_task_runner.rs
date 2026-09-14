use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use futures::TryStreamExt;
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::domain::{
    external::{
        Browser, FileStorage, JsonParser, Llm, Sandbox, SearchEngine, SharedTask, Task, TaskRunner,
        UploadFile,
    },
    models::{
        A2aConfig, A2aToolContent, AgentConfig, BrowserToolContent, DoneEvent, ErrorEvent, Event,
        File, FileToolContent, McpConfig, McpToolContent, Message, MessageEvent, SearchResults,
        SearchToolContent, SessionStatus, ShellToolContent, ToolContent, ToolEvent,
        ToolEventStatus,
    },
    repositories::{FileRepository, SessionRepository},
    services::{
        flows::{BaseFlow, PlannerReActFlow},
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
    /// 文件存储桶
    file_storage: Arc<dyn FileStorage>,
    /// 文件数据仓库
    file_repository: Arc<dyn FileRepository>,
    /// 与浏览器工具共享的浏览器实例，用于获取当前页面截图。
    browser: Arc<dyn Browser>,
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
        // 浏览器共享给流与运行器；远程工具由流持有并统一初始化、清理。
        let browser: Arc<dyn Browser> = browser.into();
        let flow = PlannerReActFlow::with_shared_browser(
            llm,
            agent_config,
            session_id.clone(),
            session_repository.clone(),
            json_parser,
            browser.clone(),
            sandbox.clone(),
            search_engine,
            McpTool::with_config(Some(mcp_config)),
            A2ATool::with_config(Some(a2a_config)),
        );

        Self {
            session_id,
            session_repository,
            sandbox,
            file_storage,
            file_repository,
            browser,
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

    /// 根据文件 id 将文件同步到沙箱中。
    async fn sync_file_to_sandbox(&self, file_id: &str) -> Option<File> {
        let result: Result<Option<File>> = async {
            // 1.调用文件存储下载文件信息。
            let (mut file_data, mut file) = self.file_storage.download_file(file_id).await?;

            // 2.组装沙箱文件路径。
            let filepath = format!("/home/ubuntu/upload/{}", file.filename);

            // 3.调用沙箱将文件上传至沙箱。
            // 存储返回异步字节流，沙箱上传接口接收完整的 Vec<u8>。
            let mut content = Vec::new();
            while let Some(chunk) = file_data.try_next().await? {
                content.extend_from_slice(&chunk);
            }
            let tool_result = self
                .sandbox
                .upload_file(content, &filepath, Some(&file.filename))
                .await?;

            // 4.判断是否上传成功。
            if tool_result.success {
                file.filepath = filepath;
                // 可以更新也可以不更新；这里保存沙箱路径，保持文件元数据同步。
                self.file_repository.save(file.clone()).await?;
                return Ok(Some(file));
            }
            Ok(None)
        }
        .await;

        match result {
            Ok(file) => file,
            Err(error) => {
                error!(file_id, error = %error, "AgentTaskRunner同步文件到沙箱失败");
                None
            }
        }
    }

    /// 将消息事件中的附件同步到沙箱中。
    async fn sync_message_attachments_to_sandbox(&self, event: &mut MessageEvent) {
        // 1.定义附件列表。
        let mut attachments = Vec::new();
        let result: Result<()> = async {
            // 2.判断消息中是否存在附件。
            if !event.attachments.is_empty() {
                // 3.循环遍历所有的消息附件。
                for attachment in &event.attachments {
                    // 4.根据同步文件的 id 将数据同步到沙箱中。
                    // 5.文件是否同步成功。
                    if let Some(file) = self.sync_file_to_sandbox(&attachment.id).await {
                        attachments.push(file.clone());
                        self.session_repository
                            .add_file(&self.session_id, file)
                            .await?;
                    }
                }

                // 6.更新消息事件中的 attachments。
                event.attachments = attachments;
            }
            Ok(())
        }
        .await;

        if let Err(error) = result {
            error!(error = %error, "AgentTaskRunner同步消息附件到沙箱失败");
        }
    }

    /// 将沙箱中指定的文件路径数据同步到存储桶中。
    async fn sync_file_to_storage(&self, filepath: &str) -> Option<File> {
        let result: Result<File> = async {
            // 1.根据文件路径从会话中查找文件数据。
            let existing_file = self
                .session_repository
                .get_file_by_path(&self.session_id, filepath)
                .await?;

            // 2.从沙箱中下载文件。
            let file_data = self.sandbox.download_file(filepath).await?;

            // 3.判断会话中的文件是否存在。
            if let Some(file) = existing_file {
                // 当前仓库按文件 id 移除记录，路径用于上一步查找。
                self.session_repository
                    .remove_file(&self.session_id, &file.id)
                    .await?;
            }

            // 4.提取文件名字、文件信息并更新文件路径。
            // UploadFile.mime_type 为 Option，允许不填写，沿用存储层的默认处理。
            let upload_file = UploadFile {
                filename: filepath.rsplit('/').next().unwrap_or_default().to_string(),
                mime_type: None,
                content: file_data.into(),
            };

            // 5.上传文件到文件存储桶。
            let mut file = self.file_storage.upload_file(upload_file).await?;
            file.filepath = filepath.to_string();

            // 6.往会话中新增一个文件信息。
            self.session_repository
                .add_file(&self.session_id, file.clone())
                .await?;
            Ok(file)
        }
        .await;

        match result {
            Ok(file) => Some(file),
            Err(error) => {
                error!(filepath, error = %error, "AgentTaskRunner同步消息附件到文件存储桶失败");
                None
            }
        }
    }

    /// 将消息事件的附件同步到文件存储桶中。
    async fn sync_message_attachments_to_storage(&self, event: &mut MessageEvent) {
        // 1.定义附件列表存储数据。
        let mut attachments = Vec::new();

        // 2.判断消息中是否存在附件；3.循环遍历所有附件。
        for attachment in &event.attachments {
            // 4.根据文件路径将数据同步到文件存储桶。
            // 单个文件的失败已记录日志，继续同步其余附件。
            if let Some(file) = self.sync_file_to_storage(&attachment.filepath).await {
                attachments.push(file);
            }
        }

        // 5.更新事件中的附件列表资源。
        event.attachments = attachments;
    }

    /// 获取浏览器截图并返回截图文件对应的 id。
    async fn get_browser_screenshot(&self) -> Result<String> {
        // 1.调用浏览器完成截图。
        let screenshot = self.browser.screenshot(None).await?;

        // 2.将浏览器截图上传到文件存储中。
        let file = self
            .file_storage
            .upload_file(UploadFile {
                filename: format!("{}.png", Uuid::new_v4()),
                mime_type: None,
                content: screenshot.into(),
            })
            .await?;
        Ok(file.id)
    }

    /// 额外处理工具消息，使其前端交互更友好。
    async fn handle_tool_event(&self, event: &mut ToolEvent) {
        // 1.如果事件状态为已调用则执行以下代码。
        if event.status != ToolEventStatus::Called {
            return;
        }

        let result: Result<()> = async {
            // 各类工具是平级分支，文件和 MCP/A2A 的处理不依赖 Shell 参数。
            match event.tool_name.as_str() {
                // 2.工具为浏览器则补全浏览器工具内容。
                "browser" => {
                    event.tool_content = Some(ToolContent::Browser(BrowserToolContent {
                        screenshot: self.get_browser_screenshot().await?,
                    }));
                }
                // 3.工具为搜索则添加搜索工具内容。
                "search" => {
                    // ToolResult<SearchResults> → data → results，展示内容只保存结果条目列表。
                    let data = event
                        .function_result
                        .as_ref()
                        .and_then(|result| result.data.as_ref())
                        .ok_or_else(|| anyhow!("搜索工具结果缺少 data"))?;
                    let search_results: SearchResults = serde_json::from_value(data.clone())?;
                    info!(count = search_results.results.len(), "处理搜索工具结果");
                    event.tool_content = Some(ToolContent::Search(SearchToolContent {
                        results: search_results.results,
                    }));
                }
                // 4.工具为 Shell 则生成 Shell 工具内容。
                "shell" => {
                    let console = if let Some(session_id) = event.function_args.get("session_id") {
                        let session_id = session_id
                            .as_str()
                            .ok_or_else(|| anyhow!("Shell session_id 必须是字符串"))?;
                        let shell_result = self
                            .sandbox
                            .read_shell_output(session_id, Some(true))
                            .await?;
                        // 现有沙箱适配器把 Shell 结构化数据编码为 JSON 字符串。
                        let data = shell_result
                            .data
                            .ok_or_else(|| anyhow!("Shell 工具结果缺少 data"))?;
                        let data: Value = serde_json::from_str(&data)?;
                        data.get("console").cloned().unwrap_or_else(|| json!([]))
                    } else {
                        json!("(No console)")
                    };
                    event.tool_content = Some(ToolContent::Shell(ShellToolContent { console }));
                }
                // 5.工具为文件则读取内容，并将文件同步到对象存储。
                "file" => {
                    if let Some(filepath) = event.function_args.get("filepath") {
                        let filepath = filepath
                            .as_str()
                            .ok_or_else(|| anyhow!("文件 filepath 必须是字符串"))?;
                        // read_file 已由适配器提取 content，data 中直接保存文件文本。
                        let file_read_result = self
                            .sandbox
                            .read_file(filepath, None, None, None, None)
                            .await?;
                        event.tool_content = Some(ToolContent::File(FileToolContent {
                            content: file_read_result.data.unwrap_or_default(),
                        }));
                        self.sync_file_to_storage(filepath).await;
                    } else {
                        event.tool_content = Some(ToolContent::File(FileToolContent {
                            content: "(No Content)".to_string(),
                        }));
                    }
                }
                // 6.工具为 MCP/A2A 则处理调用结果。
                "mcp" | "a2a" => {
                    info!(tool_name = event.tool_name, "处理MCP/A2A工具事件");
                    let result_data = match event.function_result.as_ref() {
                        // 7.如果结果包含非空 data 则提取 data。
                        Some(result) if result.data.as_ref().is_some_and(has_tool_data) => {
                            result.data.clone().unwrap()
                        }
                        // 8.MCP/A2A 工具调用正常，但是无结果产生。
                        Some(result) if result.success => serde_json::to_value(result)?,
                        // 9.其他情况将结果转换成字符串进行传递。
                        Some(result) => Value::String(serde_json::to_string(result)?),
                        None => {
                            warn!(tool_name = event.tool_name, "MCP/A2A工具调用结果未发现");
                            if event.tool_name == "mcp" {
                                json!("(MCP工具无可用结果)")
                            } else {
                                json!("(A2A智能体无可用结果)")
                            }
                        }
                    };
                    event.tool_content = Some(if event.tool_name == "mcp" {
                        ToolContent::Mcp(McpToolContent {
                            result: result_data,
                        })
                    } else {
                        ToolContent::A2a(A2aToolContent {
                            a2a_result: result_data,
                        })
                    });
                }
                _ => {}
            }
            Ok(())
        }
        .await;

        if let Err(error) = result {
            error!(tool_name = event.tool_name, error = %error, "AgentTaskRunner生成工具内容失败");
        }
    }

    /// 根据消息对象运行 PlannerReActFlow。
    async fn run_flow(&self, flow: &mut dyn BaseFlow, message: Message) -> Result<Vec<Event>> {
        // 1.判断传递的消息是否为空。
        if message.message.is_empty() {
            warn!("AgentTaskRunner接收了一条空消息");
            return Ok(vec![Event::Error(ErrorEvent {
                error: "空消息错误".to_string(),
                ..ErrorEvent::default()
            })]);
        }

        // 2.调用流并运行获取事件信息。
        // 复用现有批量事件接口；flow 由调用方持锁后借入，避免重复锁定同一实例。
        let mut events = flow.invoke(message).await?;
        // 等待事件结束当前轮次，后续事件留给用户回复后的执行。
        if let Some(index) = events
            .iter()
            .position(|event| matches!(event, Event::Wait(_)))
        {
            events.truncate(index + 1);
        }
        for event in &mut events {
            match event {
                // 3.判断是否为工具事件，如果是则额外处理。
                Event::Tool(event) => self.handle_tool_event(event).await,
                // 4.如果是消息事件则将 AI 消息事件中的附件同步到存储中。
                Event::Message(event) => self.sync_message_attachments_to_storage(event).await,
                _ => {}
            }
        }

        // 5.将事件直接返回。
        Ok(events)
    }

    /// 先写出并保存事件，再更新对应会话信息；返回是否需要等待用户。
    async fn publish_event(&self, task: &dyn Task, event: Event) -> Result<bool> {
        // 7.将得到的事件添加到消息队列中。
        self.put_and_add_event(task, event.clone()).await?;
        match event {
            // 8.如果事件类型为标题事件则更新会话标题。
            Event::Title(event) => {
                self.session_repository
                    .update_title(&self.session_id, &event.title)
                    .await?
            }
            // 9.如果事件为消息事件，则更新最新消息并新增未读消息数。
            Event::Message(event) => {
                self.session_repository
                    .update_latest_message(&self.session_id, &event.message, event.base.created_at)
                    .await?;
                self.session_repository
                    .increment_unread_message_count(&self.session_id)
                    .await?;
            }
            // 10.如果事件为等待，则更新会话状态并终止程序。
            Event::Wait(_) => {
                self.session_repository
                    .update_status(&self.session_id, SessionStatus::Waiting)
                    .await?;
                return Ok(true);
            }
            _ => {}
        }
        Ok(false)
    }
}

/// 工具结果是否包含非空数据；空集合、空文本、零和 false 使用完整结果回退。
fn has_tool_data(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
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
                let Some(event) = Self::pop_event(task.as_ref()).await? else {
                    continue;
                };

                // 4.判断事件类型是否为消息事件，如果是则处理消息并将附件同步到沙箱中。
                let Event::Message(mut event) = event else {
                    bail!("任务输入事件必须是消息事件");
                };
                self.sync_message_attachments_to_sandbox(&mut event).await;
                info!(message = %event.message.chars().take(50).collect::<String>(),
                    "AgentTaskRunner接收到新消息");

                // 5.将消息事件转换成消息对象。
                let message_obj = Message {
                    message: event.message,
                    attachments: event
                        .attachments
                        .into_iter()
                        .map(|file| file.filepath)
                        .collect(),
                };

                // 6.传递消息对象并运行 PlannerReActFlow。
                for event in self.run_flow(&mut *flow, message_obj).await? {
                    if self.publish_event(task.as_ref(), event).await? {
                        return Ok(());
                    }
                }

                // 11.判断如果输入消息队列为空则跳出循环。
                if task.input_stream().is_empty().await? {
                    break;
                }
            }

            // 12.更新会话状态为已完成。
            self.session_repository
                .update_status(&self.session_id, SessionStatus::Completed)
                .await?;
            Ok(())
        }
        .await;

        if let Err(error) = result {
            // 14.记录日志并往任务队列/消息队列中写入异常事件并更新会话状态。
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

    /// 异步任务被取消，推送结束事件并更新状态。
    async fn on_cancel(&self, task: SharedTask) -> Result<()> {
        // 13.Tokio abort 会丢弃执行 future，由任务监督器等待取消完成后调用此方法。
        info!("AgentTaskRunner任务运行取消");
        self.put_and_add_event(task.as_ref(), Event::Done(DoneEvent::default()))
            .await?;
        self.session_repository
            .update_status(&self.session_id, SessionStatus::Completed)
            .await
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
