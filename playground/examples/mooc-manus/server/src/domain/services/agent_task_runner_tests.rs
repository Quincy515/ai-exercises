use std::{
    collections::{HashMap, VecDeque},
    io,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use anyhow::{bail, Result};
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream;
use serde_json::{json, Value};
use tokio::time::timeout;

use super::AgentTaskRunner;
use crate::domain::{
    external::{
        Browser, FileStorage, FileStream, JsonParser, Llm, LlmMessage, MessageQueue, Response,
        ResponseFormat, Sandbox, SearchEngine, SharedMessageQueue, SharedTask, Task, TaskRunner,
        Tool, ToolChoice, UploadFile,
    },
    models::{
        A2aConfig, AgentConfig, DoneEvent, Event, File, FileToolContent, McpConfig, Message,
        MessageEvent, MessageRole, SearchResultItem, SearchResults, Session, SessionStatus, Step,
        TitleEvent, ToolContent, ToolEvent, ToolEventStatus, ToolResult, WaitEvent,
    },
    repositories::FileRepository,
    services::{agents::test_support::MemoryRepository, flows::BaseFlow},
};

const SESSION_ID: &str = "runner-session";
const SCREENSHOT_BYTES: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10, 0, 255, 128];

#[derive(Default)]
struct MemoryQueue {
    messages: Mutex<VecDeque<(String, Value)>>,
    next_id: AtomicUsize,
    pop_calls: AtomicUsize,
    fail_put: AtomicBool,
    fail_pop: AtomicBool,
    empty_pop_once: AtomicBool,
}

impl MemoryQueue {
    fn push(&self, id: &str, value: Value) {
        self.messages
            .lock()
            .unwrap()
            .push_back((id.to_string(), value));
    }

    fn entries(&self) -> Vec<(String, Value)> {
        self.messages.lock().unwrap().iter().cloned().collect()
    }
}

#[async_trait]
impl MessageQueue for MemoryQueue {
    async fn put(&self, message: Value) -> Result<String> {
        if self.fail_put.load(Ordering::SeqCst) {
            bail!("模拟输出队列失败");
        }
        let id = format!("{}-0", self.next_id.fetch_add(1, Ordering::SeqCst) + 1);
        self.push(&id, message);
        Ok(id)
    }

    async fn get(
        &self,
        _start_id: Option<&str>,
        _block_ms: Option<usize>,
    ) -> Result<Option<(String, Value)>> {
        panic!("运行器通过 pop 读取输入");
    }

    async fn pop(&self) -> Result<Option<(String, Value)>> {
        self.pop_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_pop.load(Ordering::SeqCst) {
            bail!("模拟输入队列失败");
        }
        if self.empty_pop_once.swap(false, Ordering::SeqCst) {
            return Ok(None);
        }
        Ok(self.messages.lock().unwrap().pop_front())
    }

    async fn clear(&self) -> Result<()> {
        panic!("本节运行器保留队列历史");
    }

    async fn is_empty(&self) -> Result<bool> {
        Ok(self.messages.lock().unwrap().is_empty())
    }

    async fn size(&self) -> Result<usize> {
        Ok(self.messages.lock().unwrap().len())
    }

    async fn delete_message(&self, _message_id: &str) -> Result<bool> {
        panic!("运行器通过 pop 移除输入");
    }
}

#[derive(Default)]
struct MemoryTask {
    input: Arc<MemoryQueue>,
    output: Arc<MemoryQueue>,
}

#[async_trait]
impl Task for MemoryTask {
    async fn invoke(&self) -> Result<()> {
        panic!("测试直接调用运行器");
    }

    fn cancel(&self) -> bool {
        panic!("本节运行器仅处理输入和事件");
    }

    fn input_stream(&self) -> SharedMessageQueue {
        self.input.clone()
    }

    fn output_stream(&self) -> SharedMessageQueue {
        self.output.clone()
    }

    fn id(&self) -> &str {
        "runner-task"
    }

    fn done(&self) -> bool {
        true
    }

    fn get(_task_id: &str) -> Result<Option<SharedTask>> {
        panic!("运行器已经收到任务实例");
    }

    async fn destroy() -> Result<()> {
        panic!("测试直接销毁运行器");
    }
}

// 依赖替身只提供固定的二进制截图；其他意外调用直接使测试失败。
struct UnusedDependency;

#[async_trait]
impl Llm for UnusedDependency {
    async fn invoke(
        &self,
        _messages: Vec<LlmMessage>,
        _tools: Option<Vec<Tool>>,
        _response_format: Option<ResponseFormat>,
        _tool_choice: Option<ToolChoice>,
    ) -> Result<Response> {
        panic!("当前测试不应调用 LLM");
    }

    fn model_name(&self) -> String {
        "unused".to_string()
    }

    fn temperature(&self) -> f32 {
        0.0
    }

    fn max_tokens(&self) -> usize {
        1024
    }
}

#[async_trait]
impl JsonParser for UnusedDependency {
    async fn invoke(&self, _text: &str, _default_value: Option<Value>) -> Result<Value> {
        panic!("本节尚未解析 LLM 输出");
    }
}

#[async_trait]
impl FileStorage for UnusedDependency {
    async fn upload_file(&self, _upload_file: UploadFile) -> Result<File> {
        panic!("本节尚未上传文件");
    }

    async fn download_file(&self, _file_id: &str) -> Result<(FileStream, File)> {
        panic!("本节尚未下载文件");
    }
}

#[async_trait]
impl FileRepository for UnusedDependency {
    async fn save(&self, _file: File) -> Result<()> {
        panic!("本节尚未保存文件");
    }

    async fn get_by_id(&self, _file_id: &str) -> Result<Option<File>> {
        panic!("本节尚未读取文件");
    }
}

#[async_trait]
impl SearchEngine for UnusedDependency {
    async fn invoke(
        &self,
        _query: String,
        _date_range: Option<String>,
    ) -> Result<ToolResult<SearchResults>> {
        panic!("本节尚未调用搜索工具");
    }
}

#[async_trait]
impl Browser for UnusedDependency {
    async fn cleanup(&self) -> Result<()> {
        panic!("浏览器资源由沙箱销毁流程处理");
    }

    async fn view_page(&self) -> Result<ToolResult<String>> {
        panic!("本节尚未调用浏览器工具");
    }

    async fn navigate(&self, _url: &str) -> Result<ToolResult<String>> {
        panic!("本节尚未调用浏览器工具");
    }

    async fn restart(&self, _url: &str) -> Result<ToolResult<String>> {
        panic!("本节尚未调用浏览器工具");
    }

    async fn click(
        &self,
        _index: Option<usize>,
        _coordinate_x: Option<f32>,
        _coordinate_y: Option<f32>,
    ) -> Result<ToolResult<String>> {
        panic!("本节尚未调用浏览器工具");
    }

    async fn input(
        &self,
        _text: &str,
        _press_enter: bool,
        _index: Option<usize>,
        _coordinate_x: Option<f32>,
        _coordinate_y: Option<f32>,
    ) -> Result<ToolResult<String>> {
        panic!("本节尚未调用浏览器工具");
    }

    async fn move_mouse(&self, _x: f32, _y: f32) -> Result<ToolResult<String>> {
        panic!("本节尚未调用浏览器工具");
    }

    async fn press_key(&self, _key: &str) -> Result<ToolResult<String>> {
        panic!("本节尚未调用浏览器工具");
    }

    async fn select_option(&self, _index: usize, _option: usize) -> Result<ToolResult<String>> {
        panic!("本节尚未调用浏览器工具");
    }

    async fn scroll_up(&self, _to_top: Option<bool>) -> Result<ToolResult<String>> {
        panic!("本节尚未调用浏览器工具");
    }

    async fn scroll_down(&self, _to_down: Option<bool>) -> Result<ToolResult<String>> {
        panic!("本节尚未调用浏览器工具");
    }

    async fn screenshot(&self, full_page: Option<bool>) -> Result<Vec<u8>> {
        assert_eq!(full_page, None);
        Ok(SCREENSHOT_BYTES.to_vec())
    }

    async fn console_exec(&self, _javascript: &str) -> Result<ToolResult<String>> {
        panic!("本节尚未调用浏览器工具");
    }

    async fn console_view(&self, _max_lines: Option<usize>) -> Result<ToolResult<String>> {
        panic!("本节尚未调用浏览器工具");
    }
}

#[derive(Default)]
struct LifecycleSandbox {
    calls: Mutex<Vec<&'static str>>,
    fail_ensure: AtomicBool,
    fail_destroy: AtomicBool,
    files: Mutex<HashMap<String, Vec<u8>>>,
    uploads: Mutex<Vec<(String, Option<String>)>>,
    fail_upload: AtomicBool,
    reject_upload: AtomicBool,
    shell_output: Mutex<Option<String>>,
    shell_reads: Mutex<Vec<(String, Option<bool>)>>,
    fail_shell_read: AtomicBool,
    file_reads: Mutex<Vec<String>>,
}

#[async_trait]
impl Sandbox for LifecycleSandbox {
    async fn exec_command(
        &self,
        _session_id: &str,
        _exec_dir: &str,
        _command: &str,
    ) -> Result<ToolResult<String>> {
        panic!("本节尚未执行沙箱命令");
    }

    async fn read_shell_output(
        &self,
        session_id: &str,
        console: Option<bool>,
    ) -> Result<ToolResult<String>> {
        self.shell_reads
            .lock()
            .unwrap()
            .push((session_id.to_string(), console));
        if self.fail_shell_read.load(Ordering::SeqCst) {
            bail!("模拟读取 Shell 输出失败");
        }
        Ok(ToolResult {
            data: self.shell_output.lock().unwrap().clone(),
            ..ToolResult::default()
        })
    }

    async fn wait_process(
        &self,
        _session_id: &str,
        _seconds: Option<usize>,
    ) -> Result<ToolResult<String>> {
        panic!("本节尚未等待沙箱进程");
    }

    async fn write_shell_input(
        &self,
        _session_id: &str,
        _input_text: &str,
        _press_enter: Option<bool>,
    ) -> Result<ToolResult<String>> {
        panic!("本节尚未写入 Shell 输入");
    }

    async fn kill_process(&self, _session_id: &str) -> Result<ToolResult<String>> {
        panic!("本节尚未管理沙箱进程");
    }

    async fn write_file(
        &self,
        _file_path: &str,
        _content: &str,
        _append: Option<bool>,
        _leading_newline: Option<bool>,
        _trailing_newline: Option<bool>,
        _sudo: Option<bool>,
    ) -> Result<ToolResult<String>> {
        panic!("本节尚未写入沙箱文件");
    }

    async fn read_file(
        &self,
        file_path: &str,
        start_line: Option<usize>,
        end_line: Option<usize>,
        sudo: Option<bool>,
        max_length: Option<usize>,
    ) -> Result<ToolResult<String>> {
        assert_eq!(
            (start_line, end_line, sudo, max_length),
            (None, None, None, None)
        );
        self.file_reads.lock().unwrap().push(file_path.to_string());
        let content = self
            .files
            .lock()
            .unwrap()
            .get(file_path)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("沙箱文件不存在"))?;
        Ok(ToolResult {
            data: Some(String::from_utf8(content)?),
            ..ToolResult::default()
        })
    }

    async fn check_file_exists(&self, _file_path: &str) -> Result<ToolResult<bool>> {
        panic!("本节尚未检查沙箱文件");
    }

    async fn delete_file(&self, _file_path: &str) -> Result<ToolResult<String>> {
        panic!("本节尚未删除沙箱文件");
    }

    async fn replace_in_file(
        &self,
        _file_path: &str,
        _old_str: &str,
        _new_str: &str,
        _sudo: Option<bool>,
    ) -> Result<ToolResult<String>> {
        panic!("本节尚未修改沙箱文件");
    }

    async fn search_in_file(
        &self,
        _file_path: &str,
        _regex: &str,
        _sudo: Option<bool>,
    ) -> Result<ToolResult<Vec<String>>> {
        panic!("本节尚未搜索沙箱文件");
    }

    async fn find_files(
        &self,
        _dir_path: &str,
        _glob_pattern: &str,
    ) -> Result<ToolResult<Vec<String>>> {
        panic!("本节尚未查找沙箱文件");
    }

    async fn upload_file(
        &self,
        file_data: Vec<u8>,
        file_path: &str,
        file_name: Option<&str>,
    ) -> Result<ToolResult<String>> {
        self.uploads
            .lock()
            .unwrap()
            .push((file_path.to_string(), file_name.map(str::to_string)));
        if self.fail_upload.load(Ordering::SeqCst) {
            bail!("模拟沙箱上传失败");
        }
        if self.reject_upload.load(Ordering::SeqCst) {
            return Ok(ToolResult::from_sandbox(500, "拒绝上传", None));
        }
        self.files
            .lock()
            .unwrap()
            .insert(file_path.to_string(), file_data);
        Ok(ToolResult::default())
    }

    async fn download_file(&self, file_path: &str) -> Result<Vec<u8>> {
        self.files
            .lock()
            .unwrap()
            .get(file_path)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("沙箱文件不存在"))
    }

    async fn ensure_sandbox(&self) -> Result<bool> {
        self.calls.lock().unwrap().push("ensure");
        if self.fail_ensure.load(Ordering::SeqCst) {
            bail!("模拟沙箱初始化失败");
        }
        Ok(true)
    }

    async fn destroy(&self) -> Result<bool> {
        self.calls.lock().unwrap().push("destroy");
        if self.fail_destroy.load(Ordering::SeqCst) {
            bail!("模拟沙箱销毁失败");
        }
        Ok(true)
    }

    async fn get_browser(&self) -> Result<Box<dyn Browser>> {
        panic!("构造函数已经接收浏览器实例");
    }

    fn id(&self) -> &str {
        "runner-sandbox"
    }

    fn cdp_url(&self) -> &str {
        ""
    }

    fn vnc_url(&self) -> &str {
        ""
    }
}

fn fixture() -> (
    AgentTaskRunner,
    Arc<MemoryRepository>,
    Arc<LifecycleSandbox>,
) {
    let repository = Arc::new(MemoryRepository::default());
    repository.insert_session(Session {
        id: SESSION_ID.to_string(),
        status: SessionStatus::Running,
        ..Session::default()
    });
    let sandbox = Arc::new(LifecycleSandbox::default());
    let runner = AgentTaskRunner::new(
        Arc::new(UnusedDependency),
        AgentConfig::default(),
        McpConfig::default(),
        A2aConfig::default(),
        SESSION_ID,
        repository.clone(),
        Arc::new(UnusedDependency),
        Arc::new(UnusedDependency),
        Arc::new(UnusedDependency),
        Box::new(UnusedDependency),
        Box::new(UnusedDependency),
        sandbox.clone(),
    );
    (runner, repository, sandbox)
}

fn message() -> Event {
    Event::Message(MessageEvent {
        role: MessageRole::User,
        message: "检查项目文件".to_string(),
        ..MessageEvent::default()
    })
}

#[tokio::test]
async fn pop_event_preserves_payload_and_uses_queue_id() {
    let task = MemoryTask::default();
    let mut expected = message();
    task.input
        .push("123-4", serde_json::to_value(&expected).unwrap());

    let event = AgentTaskRunner::pop_event(&task).await.unwrap().unwrap();

    expected.set_id("123-4");
    assert_eq!(event, expected);
    assert!(task.input.entries().is_empty());
}

#[tokio::test]
async fn pop_event_handles_empty_and_rejects_invalid_payloads() {
    let task = MemoryTask::default();
    assert!(AgentTaskRunner::pop_event(&task).await.unwrap().is_none());

    for invalid in [
        Value::Null,
        json!({"type": "unknown"}),
        json!(serde_json::to_string(&message()).unwrap()),
    ] {
        task.input.push("invalid-0", invalid);
        assert!(AgentTaskRunner::pop_event(&task).await.is_err());
    }
}

#[tokio::test]
async fn put_event_persists_queue_id_after_writing_original_payload() {
    let (runner, repository, _) = fixture();
    let task = MemoryTask::default();
    let mut event = message();
    let original_payload = serde_json::to_value(&event).unwrap();

    runner
        .put_and_add_event(&task, event.clone())
        .await
        .unwrap();

    let output = task.output.entries();
    assert_eq!(output, vec![("1-0".to_string(), original_payload)]);
    assert!(output[0].1.is_object());
    event.set_id("1-0");
    assert_eq!(repository.session(SESSION_ID).unwrap().events, vec![event]);
}

#[tokio::test]
async fn failed_output_write_prevents_persistence() {
    let (runner, repository, _) = fixture();
    let task = MemoryTask::default();
    task.output.fail_put.store(true, Ordering::SeqCst);

    let error = runner
        .put_and_add_event(&task, message())
        .await
        .unwrap_err();

    assert!(error.to_string().contains("模拟输出队列失败"));
    assert!(task.output.entries().is_empty());
    assert!(repository.session(SESSION_ID).unwrap().events.is_empty());
}

#[tokio::test]
async fn failed_persistence_keeps_the_already_queued_event() {
    let (runner, repository, _) = fixture();
    let task = MemoryTask::default();
    repository.sessions.lock().unwrap().clear();
    let event = message();

    let error = runner
        .put_and_add_event(&task, event.clone())
        .await
        .unwrap_err();

    assert!(error.to_string().contains("不存在"));
    assert_eq!(
        task.output.entries()[0].1,
        serde_json::to_value(event).unwrap()
    );
}

#[tokio::test]
async fn invoke_consumes_empty_messages_without_calling_llm() {
    let (runner, repository, sandbox) = fixture();
    let task = Arc::new(MemoryTask::default());
    for id in ["1-0", "2-0"] {
        task.input.push(
            id,
            serde_json::to_value(Event::Message(MessageEvent::default())).unwrap(),
        );
    }
    // is_empty 后的 pop 仍可能暂时为空，运行器应继续读取剩余消息。
    task.input.empty_pop_once.store(true, Ordering::SeqCst);

    runner.invoke(task.clone()).await.unwrap();

    assert_eq!(*sandbox.calls.lock().unwrap(), vec!["ensure"]);
    assert_eq!(task.input.pop_calls.load(Ordering::SeqCst), 3);
    assert!(task.input.entries().is_empty());
    assert_eq!(task.output.entries().len(), 2);
    let session = repository.session(SESSION_ID).unwrap();
    assert_eq!(session.events.len(), 2);
    assert!(session
        .events
        .iter()
        .all(|event| matches!(event, Event::Error(e) if e.error == "空消息错误")));
    assert_eq!(session.status, SessionStatus::Completed);
    assert_eq!(repository.reads.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn invoke_with_empty_input_initializes_and_returns() {
    let (runner, repository, sandbox) = fixture();
    let task = Arc::new(MemoryTask::default());

    runner.invoke(task.clone()).await.unwrap();

    assert_eq!(*sandbox.calls.lock().unwrap(), vec!["ensure"]);
    assert_eq!(task.input.pop_calls.load(Ordering::SeqCst), 0);
    assert!(task.output.entries().is_empty());
    assert_eq!(
        repository.session(SESSION_ID).unwrap().status,
        SessionStatus::Completed
    );
}

#[tokio::test]
async fn invoke_records_initialization_input_and_decoding_errors_as_completed() {
    for failure in ["ensure", "pop", "decode", "event_type"] {
        let (runner, repository, sandbox) = fixture();
        let task = Arc::new(MemoryTask::default());
        let payload = if failure == "decode" {
            json!({"type": "unknown"})
        } else if failure == "event_type" {
            serde_json::to_value(Event::Done(DoneEvent::default())).unwrap()
        } else {
            serde_json::to_value(message()).unwrap()
        };
        task.input.push("input-0", payload);
        sandbox
            .fail_ensure
            .store(failure == "ensure", Ordering::SeqCst);
        task.input
            .fail_pop
            .store(failure == "pop", Ordering::SeqCst);

        runner.invoke(task.clone()).await.unwrap();

        let session = repository.session(SESSION_ID).unwrap();
        assert_eq!(session.status, SessionStatus::Completed, "{failure}");
        assert_eq!(session.events.len(), 1, "{failure}");
        let Event::Error(error) = &session.events[0] else {
            panic!("预期错误事件: {failure}");
        };
        assert!(
            error.error.starts_with("AgentTaskRunner出错: "),
            "{failure}"
        );
        assert_eq!(error.base.id, "1-0");
        let output = task.output.entries();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].1["type"], "error");
        assert_eq!(output[0].1["error"], error.error);
        if failure == "ensure" {
            assert_eq!(task.input.pop_calls.load(Ordering::SeqCst), 0);
        }
    }
}

#[tokio::test]
async fn invoke_propagates_error_event_delivery_failure() {
    let (runner, repository, sandbox) = fixture();
    let task = Arc::new(MemoryTask::default());
    sandbox.fail_ensure.store(true, Ordering::SeqCst);
    task.output.fail_put.store(true, Ordering::SeqCst);

    let error = runner.invoke(task.clone()).await.unwrap_err();

    assert!(error.to_string().contains("模拟输出队列失败"));
    assert!(task.output.entries().is_empty());
    let session = repository.session(SESSION_ID).unwrap();
    assert!(session.events.is_empty());
    assert_eq!(session.status, SessionStatus::Running);
}

#[tokio::test]
async fn invoke_propagates_error_event_persistence_failure() {
    let (runner, repository, sandbox) = fixture();
    let task = Arc::new(MemoryTask::default());
    sandbox.fail_ensure.store(true, Ordering::SeqCst);
    repository.sessions.lock().unwrap().clear();

    let error = runner.invoke(task.clone()).await.unwrap_err();

    assert!(error.to_string().contains("不存在"));
    assert_eq!(task.output.entries().len(), 1);
    assert_eq!(task.output.entries()[0].1["type"], "error");
}

#[tokio::test]
async fn destroy_releases_sandbox_and_on_done_keeps_session_unchanged() {
    let (runner, repository, sandbox) = fixture();
    let task = Arc::new(MemoryTask::default());
    runner.invoke(task.clone()).await.unwrap();
    let before = repository.session(SESSION_ID).unwrap();

    runner.on_done(task.clone()).await.unwrap();
    assert_eq!(repository.session(SESSION_ID).unwrap(), before);
    assert!(task.output.entries().is_empty());

    runner.destroy().await.unwrap();
    assert_eq!(*sandbox.calls.lock().unwrap(), vec!["ensure", "destroy"]);
}

#[tokio::test]
async fn destroy_propagates_sandbox_failure() {
    let (runner, _, sandbox) = fixture();
    sandbox.fail_destroy.store(true, Ordering::SeqCst);

    let error = runner.destroy().await.unwrap_err();

    assert!(error.to_string().contains("模拟沙箱销毁失败"));
    assert_eq!(*sandbox.calls.lock().unwrap(), vec!["destroy"]);
}

/// 二进制存储替身；模拟存储上传自动保存元数据、下载分块读取。
#[derive(Default)]
struct MemoryFiles {
    files: Mutex<HashMap<String, (File, Vec<u8>)>>,
    uploaded: Mutex<Vec<UploadFile>>,
    saved: Mutex<Vec<File>>,
    fail_stream: AtomicBool,
    fail_save: AtomicBool,
    fail_upload: AtomicBool,
}

impl MemoryFiles {
    fn insert(&self, id: &str, filename: &str, bytes: &[u8]) -> File {
        let file = File {
            id: id.to_string(),
            filename: filename.to_string(),
            size: bytes.len(),
            ..File::default()
        };
        self.files
            .lock()
            .unwrap()
            .insert(id.to_string(), (file.clone(), bytes.to_vec()));
        file
    }
}

#[async_trait]
impl FileStorage for MemoryFiles {
    async fn upload_file(&self, upload: UploadFile) -> Result<File> {
        if self.fail_upload.load(Ordering::SeqCst) {
            bail!("模拟存储上传失败");
        }
        let file = File {
            filename: upload.filename.clone(),
            mime_type: upload.mime_type.clone().unwrap_or_default(),
            size: upload.content.len(),
            ..File::default()
        };
        self.files
            .lock()
            .unwrap()
            .insert(file.id.clone(), (file.clone(), upload.content.to_vec()));
        self.uploaded.lock().unwrap().push(upload);
        Ok(file)
    }

    async fn download_file(&self, file_id: &str) -> Result<(FileStream, File)> {
        let (file, bytes) = self
            .files
            .lock()
            .unwrap()
            .get(file_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("文件不存在"))?;
        let mut chunks: Vec<io::Result<Bytes>> = bytes
            .chunks(2)
            .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
            .collect();
        if self.fail_stream.load(Ordering::SeqCst) {
            chunks.push(Err(io::Error::other("模拟流读取失败")));
        }
        Ok((Box::pin(stream::iter(chunks)), file))
    }
}

#[async_trait]
impl FileRepository for MemoryFiles {
    async fn save(&self, file: File) -> Result<()> {
        if self.fail_save.load(Ordering::SeqCst) {
            bail!("模拟文件元数据保存失败");
        }
        self.saved.lock().unwrap().push(file.clone());
        let mut files = self.files.lock().unwrap();
        let entry = files.get_mut(&file.id).unwrap();
        entry.0 = file;
        Ok(())
    }

    async fn get_by_id(&self, file_id: &str) -> Result<Option<File>> {
        Ok(self
            .files
            .lock()
            .unwrap()
            .get(file_id)
            .map(|(file, _)| file.clone()))
    }
}

fn file_fixture() -> (
    AgentTaskRunner,
    Arc<MemoryRepository>,
    Arc<LifecycleSandbox>,
    Arc<MemoryFiles>,
) {
    let (mut runner, repository, sandbox) = fixture();
    let storage = Arc::new(MemoryFiles::default());
    runner.file_storage = storage.clone();
    runner.file_repository = storage.clone();
    (runner, repository, sandbox, storage)
}

#[tokio::test]
async fn sync_to_sandbox_collects_binary_chunks_and_saves_full_path() {
    let (runner, _, sandbox, storage) = file_fixture();
    let bytes = [0, 255, 128, 1, 2];
    storage.insert("binary", "附件.bin", &bytes);

    let file = runner.sync_file_to_sandbox("binary").await.unwrap();

    assert_eq!(file.filepath, "/home/ubuntu/upload/附件.bin");
    assert_eq!(sandbox.files.lock().unwrap()[&file.filepath], bytes);
    assert_eq!(
        *sandbox.uploads.lock().unwrap(),
        vec![(file.filepath.clone(), Some("附件.bin".to_string()))]
    );
    assert_eq!(*storage.saved.lock().unwrap(), vec![file]);
}

#[tokio::test]
async fn sync_to_sandbox_catches_download_stream_upload_and_save_failures() {
    for failure in ["missing", "stream", "upload", "rejected", "save"] {
        let (runner, _, sandbox, storage) = file_fixture();
        if failure != "missing" {
            storage.insert("file", "data.bin", &[0, 255, 1]);
        }
        storage
            .fail_stream
            .store(failure == "stream", Ordering::SeqCst);
        storage.fail_save.store(failure == "save", Ordering::SeqCst);
        sandbox
            .fail_upload
            .store(failure == "upload", Ordering::SeqCst);
        sandbox
            .reject_upload
            .store(failure == "rejected", Ordering::SeqCst);

        assert!(
            runner.sync_file_to_sandbox("file").await.is_none(),
            "{failure}"
        );
        assert!(storage.saved.lock().unwrap().is_empty(), "{failure}");
        if matches!(failure, "missing" | "stream") {
            assert!(sandbox.uploads.lock().unwrap().is_empty(), "{failure}");
        }
    }
}

#[tokio::test]
async fn input_attachments_filter_failed_files_and_register_successes() {
    let (runner, repository, _, storage) = file_fixture();
    let first = storage.insert("first", "first.txt", b"first");
    let last = storage.insert("last", "last.txt", b"last");
    let mut event = MessageEvent {
        attachments: vec![first, File::default(), last],
        ..MessageEvent::default()
    };

    runner.sync_message_attachments_to_sandbox(&mut event).await;

    assert_eq!(
        event
            .attachments
            .iter()
            .map(|file| file.id.as_str())
            .collect::<Vec<_>>(),
        vec!["first", "last"]
    );
    assert_eq!(
        repository.session(SESSION_ID).unwrap().files,
        event.attachments
    );
}

#[tokio::test]
async fn input_attachment_add_failure_keeps_original_event_and_stops_iteration() {
    let (runner, repository, sandbox, storage) = file_fixture();
    let mut event = MessageEvent {
        attachments: vec![
            storage.insert("first", "first.txt", b"1"),
            storage.insert("last", "last.txt", b"2"),
        ],
        ..MessageEvent::default()
    };
    let original = event.clone();
    repository.fail_add_file.store(true, Ordering::SeqCst);

    runner.sync_message_attachments_to_sandbox(&mut event).await;

    assert_eq!(event, original);
    assert_eq!(sandbox.uploads.lock().unwrap().len(), 1);
    assert_eq!(storage.saved.lock().unwrap().len(), 1);
    assert!(repository.session(SESSION_ID).unwrap().files.is_empty());
}

#[tokio::test]
async fn sync_to_storage_replaces_old_id_and_preserves_binary_and_missing_mime() {
    let (runner, repository, sandbox, storage) = file_fixture();
    let old = File {
        id: "old-id".to_string(),
        filepath: "/tmp/result.bin".to_string(),
        ..File::default()
    };
    let unrelated = File {
        filepath: "/tmp/keep.txt".to_string(),
        ..File::default()
    };
    repository
        .sessions
        .lock()
        .unwrap()
        .get_mut(SESSION_ID)
        .unwrap()
        .files = vec![old.clone(), unrelated.clone()];
    let bytes = vec![0, 255, 0, 128];
    sandbox
        .files
        .lock()
        .unwrap()
        .insert(old.filepath.clone(), bytes.clone());

    let file = runner.sync_file_to_storage(&old.filepath).await.unwrap();

    assert_ne!(file.id, old.id);
    assert_eq!(file.filepath, old.filepath);
    assert_eq!(
        repository.session(SESSION_ID).unwrap().files,
        vec![unrelated, file.clone()]
    );
    let uploads = storage.uploaded.lock().unwrap();
    assert_eq!(uploads.len(), 1);
    assert_eq!(uploads[0].filename, "result.bin");
    assert_eq!(uploads[0].mime_type, None);
    assert_eq!(uploads[0].content.as_ref(), bytes);
    // 本节只将沙箱路径补入事件和会话；存储元数据沿用上传时的空路径。
    assert!(storage.files.lock().unwrap()[&file.id]
        .0
        .filepath
        .is_empty());
    assert!(storage.saved.lock().unwrap().is_empty());
}

#[tokio::test]
async fn sync_to_storage_failure_respects_download_remove_upload_add_order() {
    for failure in ["lookup", "download", "upload", "add"] {
        let (runner, repository, sandbox, storage) = file_fixture();
        let old = File {
            filepath: "/tmp/result.txt".to_string(),
            ..File::default()
        };
        repository
            .sessions
            .lock()
            .unwrap()
            .get_mut(SESSION_ID)
            .unwrap()
            .files = vec![old.clone()];
        if failure == "lookup" {
            repository.sessions.lock().unwrap().clear();
        }
        if failure != "download" {
            sandbox
                .files
                .lock()
                .unwrap()
                .insert(old.filepath.clone(), b"result".to_vec());
        }
        storage
            .fail_upload
            .store(failure == "upload", Ordering::SeqCst);
        repository
            .fail_add_file
            .store(failure == "add", Ordering::SeqCst);

        assert!(
            runner.sync_file_to_storage(&old.filepath).await.is_none(),
            "{failure}"
        );

        match failure {
            "lookup" => assert!(storage.uploaded.lock().unwrap().is_empty()),
            "download" => assert_eq!(repository.session(SESSION_ID).unwrap().files, vec![old]),
            "upload" | "add" => assert!(repository.session(SESSION_ID).unwrap().files.is_empty()),
            _ => unreachable!(),
        }
        assert_eq!(
            storage.uploaded.lock().unwrap().len(),
            usize::from(failure == "add")
        );
    }
}

#[tokio::test]
async fn output_attachments_filter_failures_preserve_order_and_handle_empty_lists() {
    let (runner, repository, sandbox, _) = file_fixture();
    for path in ["/tmp/first.txt", "/tmp/last.txt"] {
        sandbox
            .files
            .lock()
            .unwrap()
            .insert(path.to_string(), path.as_bytes().to_vec());
    }
    let mut event = MessageEvent {
        attachments: ["/tmp/first.txt", "/tmp/missing.txt", "/tmp/last.txt"]
            .into_iter()
            .map(|path| File {
                filepath: path.to_string(),
                ..File::default()
            })
            .collect(),
        ..MessageEvent::default()
    };

    runner.sync_message_attachments_to_storage(&mut event).await;

    assert_eq!(
        event
            .attachments
            .iter()
            .map(|file| file.filepath.as_str())
            .collect::<Vec<_>>(),
        vec!["/tmp/first.txt", "/tmp/last.txt"]
    );
    assert_eq!(
        repository.session(SESSION_ID).unwrap().files,
        event.attachments
    );
    let mut empty = MessageEvent::default();
    runner.sync_message_attachments_to_sandbox(&mut empty).await;
    runner.sync_message_attachments_to_storage(&mut empty).await;
    assert!(empty.attachments.is_empty());
    assert_eq!(repository.session(SESSION_ID).unwrap().files.len(), 2);
}

#[derive(Default)]
struct RecordingFlow {
    messages: Vec<Message>,
    events: Vec<Event>,
    fail: bool,
}

#[async_trait]
impl BaseFlow for RecordingFlow {
    async fn invoke(&mut self, message: Message) -> Result<Vec<Event>> {
        self.messages.push(message);
        if self.fail {
            bail!("模拟 Flow 失败");
        }
        Ok(self.events.clone())
    }

    fn done(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn run_flow_rejects_only_empty_text_and_propagates_flow_errors() {
    let (runner, _, _) = fixture();
    let mut flow = RecordingFlow::default();
    let events = runner
        .run_flow(&mut flow, Message::default())
        .await
        .unwrap();
    assert!(matches!(&events[..], [Event::Error(error)] if error.error == "空消息错误"));
    assert!(flow.messages.is_empty());

    let whitespace = Message {
        message: "  ".to_string(),
        ..Message::default()
    };
    runner
        .run_flow(&mut flow, whitespace.clone())
        .await
        .unwrap();
    assert_eq!(flow.messages, vec![whitespace.clone()]);
    flow.fail = true;
    assert!(runner
        .run_flow(&mut flow, whitespace)
        .await
        .unwrap_err()
        .to_string()
        .contains("模拟 Flow 失败"));
}

#[tokio::test]
async fn run_flow_syncs_message_attachments_and_preserves_other_events() {
    let (runner, repository, sandbox, _) = file_fixture();
    sandbox
        .files
        .lock()
        .unwrap()
        .insert("/tmp/answer.txt".to_string(), b"answer".to_vec());
    let tool = Event::Tool(ToolEvent::default());
    let done = Event::Done(DoneEvent::default());
    let mut flow = RecordingFlow {
        events: vec![
            tool.clone(),
            Event::Message(MessageEvent {
                message: "文件已生成".to_string(),
                attachments: vec![File {
                    filepath: "/tmp/answer.txt".to_string(),
                    ..File::default()
                }],
                ..MessageEvent::default()
            }),
            done.clone(),
        ],
        ..RecordingFlow::default()
    };
    let input = Message {
        message: "生成文件".to_string(),
        attachments: vec!["/home/ubuntu/upload/source.txt".to_string()],
    };

    let events = runner.run_flow(&mut flow, input.clone()).await.unwrap();

    assert_eq!(flow.messages, vec![input]);
    assert_eq!(events[0], tool);
    assert_eq!(events[2], done);
    let Event::Message(message) = &events[1] else {
        panic!("应保留消息事件");
    };
    assert_eq!(message.message, "文件已生成");
    assert_eq!(
        message.attachments,
        repository.session(SESSION_ID).unwrap().files
    );
    assert_eq!(message.attachments[0].filename, "answer.txt");
}

#[derive(Default)]
struct PlanningLlm {
    requests: Mutex<Vec<Vec<LlmMessage>>>,
}

#[async_trait]
impl Llm for PlanningLlm {
    async fn invoke(
        &self,
        messages: Vec<LlmMessage>,
        _tools: Option<Vec<Tool>>,
        _response_format: Option<ResponseFormat>,
        _tool_choice: Option<ToolChoice>,
    ) -> Result<Response> {
        self.requests.lock().unwrap().push(messages);
        Ok(Response::from_iter([
            ("role".to_string(), json!("assistant")),
            (
                "content".to_string(),
                json!(json!({
                    "id": "empty-plan", "title": "检查附件", "goal": "读取附件",
                    "language": "中文", "message": "附件已接收", "steps": []
                })
                .to_string()),
            ),
        ]))
    }

    fn model_name(&self) -> String {
        "planning-test".to_string()
    }
    fn temperature(&self) -> f32 {
        0.0
    }
    fn max_tokens(&self) -> usize {
        1024
    }
}

struct TestJsonParser;

#[async_trait]
impl JsonParser for TestJsonParser {
    async fn invoke(&self, text: &str, _default_value: Option<Value>) -> Result<Value> {
        Ok(serde_json::from_str(text)?)
    }
}

#[tokio::test]
async fn invoke_runs_real_flow_with_synced_attachment_paths_without_deadlock() {
    let (_, repository, sandbox, storage) = file_fixture();
    let attachment = storage.insert("source", "材料.txt", b"source data");
    let llm = Arc::new(PlanningLlm::default());
    let runner = AgentTaskRunner::new(
        llm.clone(),
        AgentConfig {
            max_retries: 1,
            ..AgentConfig::default()
        },
        McpConfig::default(),
        A2aConfig::default(),
        SESSION_ID,
        repository.clone(),
        storage.clone(),
        storage,
        Arc::new(TestJsonParser),
        Box::new(UnusedDependency),
        Box::new(UnusedDependency),
        sandbox.clone(),
    );
    let task = Arc::new(MemoryTask::default());
    task.input.push(
        "input-1",
        serde_json::to_value(Event::Message(MessageEvent {
            message: "检查我的附件".to_string(),
            role: MessageRole::User,
            attachments: vec![attachment],
            ..MessageEvent::default()
        }))
        .unwrap(),
    );

    timeout(Duration::from_secs(2), runner.invoke(task.clone()))
        .await
        .expect("运行器与 Flow 应完成并释放锁")
        .unwrap();

    let requests = llm.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let prompt = serde_json::to_string(&requests[0]).unwrap();
    assert!(prompt.contains("检查我的附件"));
    assert!(prompt.contains("/home/ubuntu/upload/材料.txt"));
    assert!(task.input.entries().is_empty());
    let output = task.output.entries();
    assert_eq!(output.last().unwrap().1["type"], "done");
    let session = repository.session(SESSION_ID).unwrap();
    assert_eq!(session.events.len(), output.len());
    assert_eq!(session.title, "检查附件");
    assert_eq!(session.latest_message, "附件已接收");
    assert_eq!(session.unread_message_count, 1);
    assert_eq!(session.status, SessionStatus::Completed);
    assert_eq!(session.files.len(), 1);
    assert_eq!(session.files[0].filepath, "/home/ubuntu/upload/材料.txt");
    assert_eq!(*sandbox.calls.lock().unwrap(), vec!["ensure"]);
    assert!(repository.writes.load(Ordering::SeqCst) > 0);
}

fn called_tool(name: &str) -> ToolEvent {
    ToolEvent {
        tool_name: name.to_string(),
        status: ToolEventStatus::Called,
        ..ToolEvent::default()
    }
}

#[tokio::test]
async fn tool_content_is_added_only_after_call_and_screenshot_is_uploaded() {
    let (runner, repository, sandbox, storage) = file_fixture();
    for name in ["browser", "search", "shell", "file", "mcp", "a2a"] {
        let mut event = ToolEvent {
            tool_name: name.into(),
            ..ToolEvent::default()
        };
        runner.handle_tool_event(&mut event).await;
        assert!(event.tool_content.is_none());
    }
    assert!(storage.uploaded.lock().unwrap().is_empty());
    assert!(sandbox.shell_reads.lock().unwrap().is_empty());
    let mut event = called_tool("browser");
    runner.handle_tool_event(&mut event).await;
    let Some(ToolContent::Browser(content)) = event.tool_content else {
        panic!("应生成浏览器内容")
    };
    let uploads = storage.uploaded.lock().unwrap();
    assert_eq!(uploads.len(), 1);
    uuid::Uuid::parse_str(uploads[0].filename.strip_suffix(".png").unwrap()).unwrap();
    assert_eq!(uploads[0].content.as_ref(), SCREENSHOT_BYTES);
    assert_eq!(uploads[0].mime_type, None);
    assert_eq!(
        storage.files.lock().unwrap()[&content.screenshot]
            .0
            .filename,
        uploads[0].filename
    );
    assert!(repository.session(SESSION_ID).unwrap().files.is_empty());
}

#[tokio::test]
async fn search_shell_and_file_content_follow_existing_tool_contracts() {
    let (runner, repository, sandbox, storage) = file_fixture();
    let results = vec![SearchResultItem {
        url: "https://example.com".into(),
        title: "资料".into(),
        snippet: "摘要".into(),
    }];
    let mut search = called_tool("search");
    search.function_result = Some(ToolResult {
        data: Some(
            serde_json::to_value(SearchResults {
                results: results.clone(),
                ..SearchResults::default()
            })
            .unwrap(),
        ),
        ..ToolResult::default()
    });
    runner.handle_tool_event(&mut search).await;
    assert!(matches!(search.tool_content, Some(ToolContent::Search(c)) if c.results == results));

    let mut shell = called_tool("shell");
    runner.handle_tool_event(&mut shell).await;
    assert!(
        matches!(&shell.tool_content, Some(ToolContent::Shell(c)) if c.console == json!("(No console)"))
    );
    shell
        .function_args
        .insert("session_id".into(), json!("shell-1"));
    for data in [json!({"console": [{"output": "完成"}]}), json!({})] {
        *sandbox.shell_output.lock().unwrap() = Some(data.to_string());
        runner.handle_tool_event(&mut shell).await;
        let expected = data.get("console").cloned().unwrap_or(json!([]));
        assert!(
            matches!(&shell.tool_content, Some(ToolContent::Shell(c)) if c.console == expected)
        );
    }
    assert_eq!(
        *sandbox.shell_reads.lock().unwrap(),
        vec![("shell-1".into(), Some(true)); 2]
    );

    let mut file = called_tool("file");
    runner.handle_tool_event(&mut file).await;
    assert!(
        matches!(&file.tool_content, Some(ToolContent::File(c)) if c.content == "(No Content)")
    );
    sandbox
        .files
        .lock()
        .unwrap()
        .insert("/tmp/结果.txt".into(), "文件正文\n".as_bytes().to_vec());
    file.function_args
        .insert("filepath".into(), json!("/tmp/结果.txt"));
    runner.handle_tool_event(&mut file).await;
    assert!(matches!(&file.tool_content, Some(ToolContent::File(c)) if c.content == "文件正文\n"));
    assert_eq!(storage.uploaded.lock().unwrap()[0].filename, "结果.txt");
    assert_eq!(
        repository.session(SESSION_ID).unwrap().files[0].filepath,
        "/tmp/结果.txt"
    );
    assert!(sandbox.uploads.lock().unwrap().is_empty());
}

fn remote_content(event: &ToolEvent) -> &Value {
    match event.tool_content.as_ref().unwrap() {
        ToolContent::Mcp(c) => &c.result,
        ToolContent::A2a(c) => &c.a2a_result,
        _ => panic!("应生成远程工具内容"),
    }
}

#[tokio::test]
async fn enriched_tool_content_reaches_output_queue_and_session_history() {
    let (runner, repository, sandbox, storage) = file_fixture();
    let task = MemoryTask::default();
    let filepath = "/tmp/章节总结.md";
    sandbox
        .files
        .lock()
        .unwrap()
        .insert(filepath.into(), b"report".to_vec());

    let results = json!([{"url": "https://example.com", "title": "资料", "snippet": "摘要"}]);
    let mut search = called_tool("search");
    search.function_result = Some(ToolResult {
        data: Some(json!({"query": "资料", "total_results": 1, "results": results})),
        ..ToolResult::default()
    });
    let mut file = called_tool("file");
    file.function_args
        .insert("filepath".into(), json!(filepath));
    let mut mcp = called_tool("mcp");
    mcp.function_result = Some(ToolResult {
        data: Some(json!({"answer": "完成"})),
        ..ToolResult::default()
    });

    // 同一轮包含多种工具：文件和远程工具没有 session_id，仍应生成各自的展示内容。
    let cases = [
        (search, json!({"results": results})),
        (called_tool("shell"), json!({"console": "(No console)"})),
        (file, json!({"content": "report"})),
        (mcp, json!({"result": {"answer": "完成"}})),
        (
            called_tool("a2a"),
            json!({"a2a_result": "(A2A智能体无可用结果)"}),
        ),
    ];
    let (events, expected): (Vec<_>, Vec<_>) = cases
        .into_iter()
        .map(|(event, content)| (Event::Tool(event), content))
        .unzip();
    let mut flow = RecordingFlow {
        events,
        ..RecordingFlow::default()
    };
    let events = runner
        .run_flow(
            &mut flow,
            Message {
                message: "生成章节总结".into(),
                ..Message::default()
            },
        )
        .await
        .unwrap();
    for event in events {
        assert!(!runner.publish_event(&task, event).await.unwrap());
    }

    let output = task.output.entries();
    let session = repository.session(SESSION_ID).unwrap();
    assert_eq!(output.len(), expected.len());
    assert_eq!(session.events.len(), expected.len());
    for (index, (queue_id, payload)) in output.iter().enumerate() {
        let stored = serde_json::to_value(&session.events[index]).unwrap();
        assert_eq!(payload["tool_content"], expected[index]);
        assert_eq!(stored["tool_content"], expected[index]);
        assert_eq!(stored["id"], *queue_id);
    }
    // 文件从沙箱上传到存储桶，不把沙箱路径当文件 ID 再下载回沙箱。
    assert_eq!(
        storage.uploaded.lock().unwrap()[0].content.as_ref(),
        b"report"
    );
    assert_eq!(session.files[0].filepath, filepath);
    assert!(sandbox.uploads.lock().unwrap().is_empty());
    assert!(sandbox.shell_reads.lock().unwrap().is_empty());
}

#[tokio::test]
async fn remote_results_cover_data_empty_success_failure_and_missing_result() {
    let (runner, _, _) = fixture();
    for name in ["mcp", "a2a"] {
        for success in [true, false] {
            for (data, nonempty) in [
                (Some(json!({"answer": 42})), true),
                (Some(json!(false)), false),
                (Some(json!(0)), false),
                (Some(json!("")), false),
                (Some(json!([])), false),
                (Some(json!({})), false),
                (Some(Value::Null), false),
                (None, false),
            ] {
                let result = ToolResult {
                    success,
                    message: Some("执行结果".into()),
                    data: data.clone(),
                };
                let mut event = called_tool(name);
                event.function_result = Some(result.clone());
                runner.handle_tool_event(&mut event).await;
                let content = remote_content(&event);
                if nonempty {
                    assert_eq!(content, data.as_ref().unwrap());
                } else if success {
                    assert_eq!(*content, serde_json::to_value(&result).unwrap());
                } else {
                    assert_eq!(
                        serde_json::from_str::<Value>(content.as_str().unwrap()).unwrap(),
                        serde_json::to_value(&result).unwrap()
                    );
                }
            }
        }
        let mut missing = called_tool(name);
        runner.handle_tool_event(&mut missing).await;
        assert_eq!(
            *remote_content(&missing),
            if name == "mcp" {
                json!("(MCP工具无可用结果)")
            } else {
                json!("(A2A智能体无可用结果)")
            }
        );
    }
}

#[tokio::test]
async fn tool_enrichment_failure_keeps_original_payload_and_flow_continues() {
    for name in ["browser", "search", "shell", "file"] {
        let (runner, _, sandbox, storage) = file_fixture();
        let original = Some(ToolContent::File(FileToolContent {
            content: "原始内容".into(),
        }));
        let mut event = called_tool(name);
        event.tool_content = original.clone();
        storage.fail_upload.store(true, Ordering::SeqCst);
        event
            .function_args
            .insert("session_id".into(), json!("shell-1"));
        event
            .function_args
            .insert("filepath".into(), json!("/missing.txt"));
        *sandbox.shell_output.lock().unwrap() = Some("invalid-json".into());
        event.function_result = Some(ToolResult {
            data: Some(json!("invalid-search")),
            ..ToolResult::default()
        });
        let mut flow = RecordingFlow {
            events: vec![Event::Tool(event), Event::Done(DoneEvent::default())],
            ..RecordingFlow::default()
        };
        let events = runner
            .run_flow(
                &mut flow,
                Message {
                    message: "执行".into(),
                    ..Message::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], Event::Tool(e) if e.tool_content == original));
        assert!(matches!(events[1], Event::Done(_)));
    }
}

#[tokio::test]
async fn published_events_persist_before_updating_session_metadata_and_waiting() {
    let (runner, repository, _) = fixture();
    let task = MemoryTask::default();
    let message = MessageEvent {
        message: "新的消息".into(),
        ..MessageEvent::default()
    };
    let timestamp = message.base.created_at;
    assert!(!runner
        .publish_event(
            &task,
            Event::Title(TitleEvent {
                title: "新的标题".into(),
                ..TitleEvent::default()
            })
        )
        .await
        .unwrap());
    assert!(!runner
        .publish_event(&task, Event::Message(message))
        .await
        .unwrap());
    assert!(runner
        .publish_event(&task, Event::Wait(WaitEvent::default()))
        .await
        .unwrap());
    let session = repository.session(SESSION_ID).unwrap();
    assert_eq!(session.title, "新的标题");
    assert_eq!(session.latest_message, "新的消息");
    assert_eq!(session.latest_message_at, Some(timestamp));
    assert_eq!(session.unread_message_count, 1);
    assert_eq!(session.status, SessionStatus::Waiting);
    for (event, (id, payload)) in session.events.iter().zip(task.output.entries()) {
        let mut stored = serde_json::to_value(event).unwrap();
        assert_eq!(stored["id"], id);
        stored["id"] = payload["id"].clone();
        assert_eq!(stored, payload);
    }
    assert_eq!(session.events.len(), 3);
}

#[tokio::test]
async fn publication_failure_stops_later_metadata_changes() {
    for failure in ["queue", "persistence", "metadata"] {
        let (runner, repository, _) = fixture();
        let task = MemoryTask::default();
        task.output
            .fail_put
            .store(failure == "queue", Ordering::SeqCst);
        repository
            .fail_add_event
            .store(failure == "persistence", Ordering::SeqCst);
        repository
            .fail_metadata
            .store(failure == "metadata", Ordering::SeqCst);
        assert!(runner
            .publish_event(
                &task,
                Event::Title(TitleEvent {
                    title: "新标题".into(),
                    ..TitleEvent::default()
                })
            )
            .await
            .is_err());
        let session = repository.session(SESSION_ID).unwrap();
        assert!(session.title.is_empty());
        assert_eq!(task.output.entries().len(), usize::from(failure != "queue"));
        assert_eq!(session.events.len(), usize::from(failure == "metadata"));
    }
}

#[tokio::test]
async fn cancellation_emits_done_and_completes_after_successful_delivery() {
    let (runner, repository, _) = fixture();
    let task = Arc::new(MemoryTask::default());
    runner.on_cancel(task.clone()).await.unwrap();
    let session = repository.session(SESSION_ID).unwrap();
    assert_eq!(session.status, SessionStatus::Completed);
    assert!(matches!(&session.events[..], [Event::Done(e)] if e.base.id == "1-0"));
    assert_eq!(task.output.entries()[0].1["type"], "done");
    for failure in ["queue", "persistence", "status"] {
        let (runner, repository, _) = fixture();
        let task = Arc::new(MemoryTask::default());
        task.output
            .fail_put
            .store(failure == "queue", Ordering::SeqCst);
        repository
            .fail_add_event
            .store(failure == "persistence", Ordering::SeqCst);
        repository
            .fail_update_status
            .store(failure == "status", Ordering::SeqCst);
        assert!(runner.on_cancel(task).await.is_err());
        assert_eq!(
            repository.session(SESSION_ID).unwrap().status,
            SessionStatus::Running
        );
    }
}

#[tokio::test]
async fn wait_stops_enrichment_and_side_effects_of_following_events() {
    let (runner, _, _, storage) = file_fixture();
    let mut flow = RecordingFlow {
        events: vec![
            Event::Wait(WaitEvent::default()),
            Event::Tool(called_tool("browser")),
        ],
        ..RecordingFlow::default()
    };
    let events = runner
        .run_flow(
            &mut flow,
            Message {
                message: "等待".into(),
                ..Message::default()
            },
        )
        .await
        .unwrap();
    assert!(matches!(&events[..], [Event::Wait(_)]));
    assert!(storage.uploaded.lock().unwrap().is_empty());
}

struct WaitingLlm {
    requests: AtomicUsize,
}

#[async_trait]
impl Llm for WaitingLlm {
    async fn invoke(
        &self,
        _messages: Vec<LlmMessage>,
        _tools: Option<Vec<Tool>>,
        _format: Option<ResponseFormat>,
        _choice: Option<ToolChoice>,
    ) -> Result<Response> {
        let response = if self.requests.fetch_add(1, Ordering::SeqCst) == 0 {
            json!({"role":"assistant", "content": json!({"title":"等待确认", "goal":"执行任务", "message":"请确认", "steps":[Step::new("等待用户确认")]}).to_string()})
        } else {
            json!({"role":"assistant", "content":null, "tool_calls":[{"id":"ask-1", "function":{"name":"message_ask_user", "arguments":"{\"text\":\"是否继续？\"}"}}]})
        };
        Ok(serde_json::from_value(response).unwrap())
    }
    fn model_name(&self) -> String {
        "waiting-test".into()
    }
    fn temperature(&self) -> f32 {
        0.0
    }
    fn max_tokens(&self) -> usize {
        1024
    }
}

fn runner_using_llm(
    llm: Arc<dyn Llm>,
    repository: Arc<MemoryRepository>,
    sandbox: Arc<LifecycleSandbox>,
) -> AgentTaskRunner {
    AgentTaskRunner::new(
        llm,
        AgentConfig {
            max_retries: 1,
            max_iterations: 2,
            ..AgentConfig::default()
        },
        McpConfig::default(),
        A2aConfig::default(),
        SESSION_ID,
        repository,
        Arc::new(UnusedDependency),
        Arc::new(UnusedDependency),
        Arc::new(TestJsonParser),
        Box::new(UnusedDependency),
        Box::new(UnusedDependency),
        sandbox,
    )
}

#[tokio::test]
async fn real_flow_wait_keeps_pending_input_and_normal_completion_drains_it() {
    for waiting in [true, false] {
        let (_, repository, sandbox) = fixture();
        let waiting_llm = Arc::new(WaitingLlm {
            requests: AtomicUsize::new(0),
        });
        let planning_llm = Arc::new(PlanningLlm::default());
        let llm: Arc<dyn Llm> = if waiting {
            waiting_llm.clone()
        } else {
            planning_llm.clone()
        };
        let runner = runner_using_llm(llm, repository.clone(), sandbox);
        let task = Arc::new(MemoryTask::default());
        for id in ["input-1", "input-2"] {
            task.input
                .push(id, serde_json::to_value(message()).unwrap());
        }
        timeout(Duration::from_secs(2), runner.invoke(task.clone()))
            .await
            .unwrap()
            .unwrap();
        let session = repository.session(SESSION_ID).unwrap();
        let output = task.output.entries();
        assert_eq!(session.events.len(), output.len());
        if waiting {
            assert_eq!(session.status, SessionStatus::Waiting);
            assert_eq!(task.input.entries().len(), 1);
            assert_eq!(output.last().unwrap().1["type"], "wait");
            assert!(output.iter().all(|(_, event)| event["type"] != "done"));
            assert_eq!(waiting_llm.requests.load(Ordering::SeqCst), 2);
        } else {
            assert_eq!(session.status, SessionStatus::Completed);
            assert!(task.input.entries().is_empty());
            assert_eq!(
                output
                    .iter()
                    .filter(|(_, event)| event["type"] == "done")
                    .count(),
                2
            );
            assert_eq!(planning_llm.requests.lock().unwrap().len(), 2);
            assert_eq!(session.unread_message_count, 2);
        }
    }
}
