use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::domain::{
    external::Sandbox,
    models::ToolResult,
    services::tools::{tool, BaseTool, ToolArguments, ToolDefinition},
};

/// Shell 工具箱，提供 Shell 交互相关功能。
/// Shell toolbox for executing and interacting with sandbox shell sessions.
pub struct ShellTool {
    name: String,
    sandbox: Box<dyn Sandbox>,
    definitions: Vec<ToolDefinition>,
}

impl ShellTool {
    /// 构造函数，完成 Shell 工具箱初始化。
    /// Create a shell tool collection with a sandbox backend.
    pub fn new(sandbox: Box<dyn Sandbox>) -> Self {
        Self {
            name: "shell".to_string(),
            sandbox,
            definitions: vec![
                tool(
                    "shell_execute",
                    "在指定 Shell 会话中执行命令。可用于运行代码、安装依赖包或文件管理。",
                    ToolArguments::from_iter([
                        (
                            "session_id".to_string(),
                            json!({
                                "type": "string",
                                "description": "目标 Shell 会话的唯一标识符"
                            }),
                        ),
                        (
                            "exec_dir".to_string(),
                            json!({
                                "type": "string",
                                "description": "执行命令的工作目录（必须使用绝对路径）"
                            }),
                        ),
                        (
                            "command".to_string(),
                            json!({
                                "type": "string",
                                "description": "要执行的 Shell 命令"
                            }),
                        ),
                    ]),
                    vec![
                        "session_id".to_string(),
                        "exec_dir".to_string(),
                        "command".to_string(),
                    ],
                ),
                tool(
                    "shell_view",
                    "查看指定 Shell 会话的内容。用于检查命令执行结果或监控输出。",
                    ToolArguments::from_iter([(
                        "session_id".to_string(),
                        json!({
                            "type": "string",
                            "description": "目标 Shell 会话的唯一标识符"
                        }),
                    )]),
                    vec!["session_id".to_string()],
                ),
                tool(
                    "shell_wait",
                    "等待指定 Shell 会话中正在运行的进程返回。在运行耗时较长的命令后使用。",
                    ToolArguments::from_iter([
                        (
                            "session_id".to_string(),
                            json!({
                                "type": "string",
                                "description": "目标 Shell 会话的唯一标识符"
                            }),
                        ),
                        (
                            "seconds".to_string(),
                            json!({
                                "type": "integer",
                                "description": "可选参数, 等待时长（秒）"
                            }),
                        ),
                    ]),
                    vec!["session_id".to_string()],
                ),
                tool(
                    "shell_write_to_process",
                    "向指定 Shell 会话中正在运行的进程写入输入。用于响应交互式命令提示符。",
                    ToolArguments::from_iter([
                        (
                            "session_id".to_string(),
                            json!({
                                "type": "string",
                                "description": "目标 Shell 会话的唯一标识符"
                            }),
                        ),
                        (
                            "input_text".to_string(),
                            json!({
                                "type": "string",
                                "description": "要写入进程的输入内容"
                            }),
                        ),
                        (
                            "press_enter".to_string(),
                            json!({
                                "type": "boolean",
                                "description": "输入后是否按下回车键"
                            }),
                        ),
                    ]),
                    vec![
                        "session_id".to_string(),
                        "input_text".to_string(),
                        "press_enter".to_string(),
                    ],
                ),
                tool(
                    "shell_kill_process",
                    "在指定 Shell 会话中终止正在运行的进程。用于停止长时间运行的进程或处理卡死的命令。",
                    ToolArguments::from_iter([(
                        "session_id".to_string(),
                        json!({
                            "type": "string",
                            "description": "目标 Shell 会话的唯一标识符"
                        }),
                    )]),
                    vec!["session_id".to_string()],
                ),
            ],
        }
    }

    /// 执行 Shell 命令。
    /// Execute a shell command in a sandbox session.
    async fn shell_execute(
        &self,
        session_id: &str,
        exec_dir: &str,
        command: &str,
    ) -> Result<ToolResult<String>> {
        self.sandbox
            .exec_command(session_id, exec_dir, command)
            .await
    }

    /// 根据会话 id 查看 Shell 会话内容。
    /// Read output from a shell session.
    async fn shell_view(&self, session_id: &str) -> Result<ToolResult<String>> {
        self.sandbox.view_shell(session_id, Some(false)).await
    }

    /// 等待指定 Shell 会话中正在运行的进程返回。
    /// Wait for the running process in a shell session.
    async fn shell_wait(
        &self,
        session_id: &str,
        seconds: Option<usize>,
    ) -> Result<ToolResult<String>> {
        self.sandbox.wait_for_process(session_id, seconds).await
    }

    /// 向指定 Shell 会话正在运行的进程写入输入。
    /// Write input to the running process in a shell session.
    async fn shell_write_to_process(
        &self,
        session_id: &str,
        input_text: &str,
        press_enter: bool,
    ) -> Result<ToolResult<String>> {
        self.sandbox
            .write_to_process(session_id, input_text, Some(press_enter))
            .await
    }

    /// 在指定 Shell 会话中终止正在运行的进程。
    /// Kill the running process in a shell session.
    async fn shell_kill_process(&self, session_id: &str) -> Result<ToolResult<String>> {
        self.sandbox.kill_process(session_id).await
    }
}

#[async_trait]
impl BaseTool for ShellTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn tool_definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    async fn call_tool(&self, tool_name: &str, kwargs: ToolArguments) -> Result<ToolResult<Value>> {
        let result = match tool_name {
            "shell_execute" => {
                let session_id = required_str(&kwargs, "session_id")?;
                let exec_dir = required_str(&kwargs, "exec_dir")?;
                let command = required_str(&kwargs, "command")?;
                self.shell_execute(session_id, exec_dir, command).await?
            }
            "shell_view" => {
                let session_id = required_str(&kwargs, "session_id")?;
                self.shell_view(session_id).await?
            }
            "shell_wait" => {
                let session_id = required_str(&kwargs, "session_id")?;
                let seconds = optional_usize(&kwargs, "seconds")?;
                self.shell_wait(session_id, seconds).await?
            }
            "shell_write_to_process" => {
                let session_id = required_str(&kwargs, "session_id")?;
                let input_text = required_str(&kwargs, "input_text")?;
                let press_enter = required_bool(&kwargs, "press_enter")?;
                self.shell_write_to_process(session_id, input_text, press_enter)
                    .await?
            }
            "shell_kill_process" => {
                let session_id = required_str(&kwargs, "session_id")?;
                self.shell_kill_process(session_id).await?
            }
            _ => return Err(anyhow!("工具[{tool_name}]未找到")),
        };

        Ok(ToolResult {
            success: result.success,
            message: result.message,
            data: result.data.map(Value::String),
        })
    }
}

fn required_str<'a>(kwargs: &'a ToolArguments, name: &str) -> Result<&'a str> {
    kwargs
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("工具参数[{name}]缺失"))
}

fn required_bool(kwargs: &ToolArguments, name: &str) -> Result<bool> {
    kwargs
        .get(name)
        .and_then(Value::as_bool)
        .ok_or_else(|| anyhow!("工具参数[{name}]缺失"))
}

fn optional_usize(kwargs: &ToolArguments, name: &str) -> Result<Option<usize>> {
    match kwargs.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .map(Some)
            .ok_or_else(|| anyhow!("工具参数[{name}]必须是非负整数")),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use anyhow::bail;
    use serde_json::{json, Map};

    use super::*;
    use crate::domain::external::Browser;

    type Calls = Arc<Mutex<Vec<ShellCall>>>;

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ShellCall {
        Execute {
            session_id: String,
            exec_dir: String,
            command: String,
        },
        ReadOutput {
            session_id: String,
            console: Option<bool>,
        },
        WaitProcess {
            session_id: String,
            seconds: Option<usize>,
        },
        WriteInput {
            session_id: String,
            input_text: String,
            press_enter: Option<bool>,
        },
        KillProcess {
            session_id: String,
        },
    }

    struct MockSandbox {
        calls: Calls,
    }

    impl MockSandbox {
        fn record(&self, call: ShellCall, data: &str) -> ToolResult<String> {
            self.calls.lock().unwrap().push(call);
            ToolResult {
                data: Some(data.to_string()),
                ..ToolResult::default()
            }
        }
    }

    #[async_trait]
    impl Sandbox for MockSandbox {
        async fn exec_command(
            &self,
            session_id: &str,
            exec_dir: &str,
            command: &str,
        ) -> Result<ToolResult<String>> {
            Ok(self.record(
                ShellCall::Execute {
                    session_id: session_id.to_string(),
                    exec_dir: exec_dir.to_string(),
                    command: command.to_string(),
                },
                "execute",
            ))
        }

        async fn view_shell(
            &self,
            session_id: &str,
            console: Option<bool>,
        ) -> Result<ToolResult<String>> {
            Ok(self.record(
                ShellCall::ReadOutput {
                    session_id: session_id.to_string(),
                    console,
                },
                "read",
            ))
        }

        async fn wait_for_process(
            &self,
            session_id: &str,
            seconds: Option<usize>,
        ) -> Result<ToolResult<String>> {
            Ok(self.record(
                ShellCall::WaitProcess {
                    session_id: session_id.to_string(),
                    seconds,
                },
                "wait",
            ))
        }

        async fn write_to_process(
            &self,
            session_id: &str,
            input_text: &str,
            press_enter: Option<bool>,
        ) -> Result<ToolResult<String>> {
            Ok(self.record(
                ShellCall::WriteInput {
                    session_id: session_id.to_string(),
                    input_text: input_text.to_string(),
                    press_enter,
                },
                "write",
            ))
        }

        async fn kill_process(&self, session_id: &str) -> Result<ToolResult<String>> {
            Ok(self.record(
                ShellCall::KillProcess {
                    session_id: session_id.to_string(),
                },
                "kill",
            ))
        }

        async fn file_write(
            &self,
            _file_path: &str,
            _content: &str,
            _append: Option<bool>,
            _leading_newline: Option<bool>,
            _trailing_newline: Option<bool>,
            _sudo: Option<bool>,
        ) -> Result<ToolResult<String>> {
            bail!("file_write should not be called by ShellTool")
        }

        async fn file_read(
            &self,
            _file_path: &str,
            _start_line: Option<usize>,
            _end_line: Option<usize>,
            _sudo: Option<bool>,
        ) -> Result<ToolResult<String>> {
            bail!("file_read should not be called by ShellTool")
        }

        async fn file_exists(&self, _file_path: &str) -> Result<ToolResult<bool>> {
            bail!("file_exists should not be called by ShellTool")
        }

        async fn file_delete(&self, _file_path: &str) -> Result<ToolResult<String>> {
            bail!("file_delete should not be called by ShellTool")
        }

        async fn file_list(&self, _dir_path: &str) -> Result<ToolResult<Vec<String>>> {
            bail!("file_list should not be called by ShellTool")
        }

        async fn file_replace(
            &self,
            _file_path: &str,
            _old_str: &str,
            _new_str: &str,
            _sudo: Option<bool>,
        ) -> Result<ToolResult<String>> {
            bail!("file_replace should not be called by ShellTool")
        }

        async fn file_search(
            &self,
            _file_path: &str,
            _regex: &str,
            _sudo: Option<bool>,
        ) -> Result<ToolResult<Vec<String>>> {
            bail!("file_search should not be called by ShellTool")
        }

        async fn file_find(
            &self,
            _dir_path: &str,
            _glob_pattern: &str,
        ) -> Result<ToolResult<Vec<String>>> {
            bail!("file_find should not be called by ShellTool")
        }

        async fn file_upload(
            &self,
            _file_data: Vec<u8>,
            _file_path: &str,
            _file_name: Option<&str>,
        ) -> Result<ToolResult<String>> {
            bail!("file_upload should not be called by ShellTool")
        }

        async fn file_download(&self, _file_path: &str) -> Result<Vec<u8>> {
            bail!("file_download should not be called by ShellTool")
        }

        async fn ensure_sandbox(&self) -> Result<bool> {
            bail!("ensure_sandbox should not be called by ShellTool")
        }

        async fn destroy(&self) -> Result<bool> {
            bail!("destroy should not be called by ShellTool")
        }

        async fn get_browser(&self) -> Result<Box<dyn Browser>> {
            bail!("get_browser should not be called by ShellTool")
        }

        fn id(&self) -> &str {
            "mock-sandbox"
        }

        fn cdp_url(&self) -> &str {
            "ws://mock-cdp"
        }

        fn vnc_url(&self) -> &str {
            "ws://mock-vnc"
        }
    }

    fn shell_tool() -> (ShellTool, Calls) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let sandbox = MockSandbox {
            calls: Arc::clone(&calls),
        };

        (ShellTool::new(Box::new(sandbox)), calls)
    }

    #[test]
    fn exposes_shell_schema_like_python_tool_decorator() {
        let (tool, _) = shell_tool();

        assert_eq!(tool.name(), "shell");
        assert!(tool.has_tool("shell_execute"));
        assert!(tool.has_tool("shell_view"));
        assert!(!tool.has_tool("missing"));

        let tools = tool.get_tools();
        let tool_names = tools
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            tool_names,
            vec![
                "shell_execute",
                "shell_view",
                "shell_wait",
                "shell_write_to_process",
                "shell_kill_process",
            ]
        );
        assert_eq!(
            tools[0]["function"]["parameters"]["required"],
            json!(["session_id", "exec_dir", "command"])
        );
        assert_eq!(
            tools[3]["function"]["parameters"]["required"],
            json!(["session_id", "input_text", "press_enter"])
        );
    }

    #[tokio::test]
    async fn dispatches_python_shell_tools_to_sandbox() {
        let (tool, calls) = shell_tool();

        assert_eq!(
            tool.invoke(
                "shell_execute",
                Map::from_iter([
                    ("session_id".to_string(), json!("s1")),
                    ("exec_dir".to_string(), json!("/workspace")),
                    ("command".to_string(), json!("cargo test")),
                ]),
            )
            .await
            .unwrap()
            .data
            .unwrap(),
            "execute"
        );
        tool.invoke(
            "shell_view",
            Map::from_iter([("session_id".to_string(), json!("s1"))]),
        )
        .await
        .unwrap();
        tool.invoke(
            "shell_wait",
            Map::from_iter([
                ("session_id".to_string(), json!("s1")),
                ("seconds".to_string(), json!(30)),
            ]),
        )
        .await
        .unwrap();
        tool.invoke(
            "shell_write_to_process",
            Map::from_iter([
                ("session_id".to_string(), json!("s1")),
                ("input_text".to_string(), json!("y")),
                ("press_enter".to_string(), json!(true)),
            ]),
        )
        .await
        .unwrap();
        tool.invoke(
            "shell_kill_process",
            Map::from_iter([("session_id".to_string(), json!("s1"))]),
        )
        .await
        .unwrap();

        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                ShellCall::Execute {
                    session_id: "s1".to_string(),
                    exec_dir: "/workspace".to_string(),
                    command: "cargo test".to_string(),
                },
                ShellCall::ReadOutput {
                    session_id: "s1".to_string(),
                    console: Some(false),
                },
                ShellCall::WaitProcess {
                    session_id: "s1".to_string(),
                    seconds: Some(30),
                },
                ShellCall::WriteInput {
                    session_id: "s1".to_string(),
                    input_text: "y".to_string(),
                    press_enter: Some(true),
                },
                ShellCall::KillProcess {
                    session_id: "s1".to_string(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn shell_wait_process_accepts_missing_seconds() {
        let (tool, calls) = shell_tool();

        let result = tool
            .invoke(
                "shell_wait",
                Map::from_iter([("session_id".to_string(), json!("s2"))]),
            )
            .await
            .unwrap();

        assert_eq!(result.data.unwrap(), "wait");
        assert_eq!(
            *calls.lock().unwrap(),
            vec![ShellCall::WaitProcess {
                session_id: "s2".to_string(),
                seconds: None,
            }]
        );
    }

    #[tokio::test]
    async fn rejects_missing_required_shell_argument() {
        let (tool, _) = shell_tool();
        let error = tool
            .invoke(
                "shell_execute",
                Map::from_iter([
                    ("session_id".to_string(), json!("s1")),
                    ("exec_dir".to_string(), json!("/workspace")),
                ]),
            )
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "工具参数[command]缺失");
    }

    #[tokio::test]
    async fn rejects_invalid_optional_seconds() {
        let (tool, _) = shell_tool();
        let error = tool
            .invoke(
                "shell_wait",
                Map::from_iter([
                    ("session_id".to_string(), json!("s1")),
                    ("seconds".to_string(), json!(-1)),
                ]),
            )
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "工具参数[seconds]必须是非负整数");
    }
}
