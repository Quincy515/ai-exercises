use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use anyhow::{bail, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use super::AgentTaskRunner;
use crate::domain::{
    external::{
        Browser, FileStorage, FileStream, JsonParser, Llm, LlmMessage, MessageQueue, Response,
        ResponseFormat, Sandbox, SearchEngine, SharedMessageQueue, SharedTask, Task, TaskRunner,
        Tool, ToolChoice, UploadFile,
    },
    models::{
        A2aConfig, AgentConfig, Event, File, McpConfig, MessageEvent, MessageRole, SearchResults,
        Session, SessionStatus, ToolResult,
    },
    repositories::FileRepository,
    services::agents::test_support::MemoryRepository,
};

const SESSION_ID: &str = "runner-session";

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

// 本节保留 Flow 执行和文件同步的 TODO；调用这些依赖会直接使测试失败。
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
        panic!("本节尚未执行 Flow 或调用 LLM");
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

    async fn screenshot(&self, _full_page: Option<bool>) -> Result<Vec<u8>> {
        panic!("本节尚未调用浏览器工具");
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
        _session_id: &str,
        _console: Option<bool>,
    ) -> Result<ToolResult<String>> {
        panic!("本节尚未读取 Shell 输出");
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
        _file_path: &str,
        _start_line: Option<usize>,
        _end_line: Option<usize>,
        _sudo: Option<bool>,
        _max_length: Option<usize>,
    ) -> Result<ToolResult<String>> {
        panic!("本节尚未读取沙箱文件");
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
        _file_data: Vec<u8>,
        _file_path: &str,
        _file_name: Option<&str>,
    ) -> Result<ToolResult<String>> {
        panic!("本节尚未上传沙箱文件");
    }

    async fn download_file(&self, _file_path: &str) -> Result<Vec<u8>> {
        panic!("本节尚未下载沙箱文件");
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
async fn invoke_initializes_and_consumes_input_with_flow_execution_pending() {
    let (runner, repository, sandbox) = fixture();
    let task = Arc::new(MemoryTask::default());
    for id in ["1-0", "2-0"] {
        task.input
            .push(id, serde_json::to_value(message()).unwrap());
    }
    // is_empty 后的 pop 仍可能暂时为空，运行器应继续读取剩余消息。
    task.input.empty_pop_once.store(true, Ordering::SeqCst);

    runner.invoke(task.clone()).await.unwrap();

    assert_eq!(*sandbox.calls.lock().unwrap(), vec!["ensure"]);
    assert_eq!(task.input.pop_calls.load(Ordering::SeqCst), 3);
    assert!(task.input.entries().is_empty());
    assert!(task.output.entries().is_empty());
    let session = repository.session(SESSION_ID).unwrap();
    assert!(session.events.is_empty());
    assert_eq!(session.status, SessionStatus::Running);
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
        SessionStatus::Running
    );
}

#[tokio::test]
async fn invoke_records_initialization_input_and_decoding_errors_as_completed() {
    for failure in ["ensure", "pop", "decode"] {
        let (runner, repository, sandbox) = fixture();
        let task = Arc::new(MemoryTask::default());
        let payload = if failure == "decode" {
            json!({"type": "unknown"})
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
