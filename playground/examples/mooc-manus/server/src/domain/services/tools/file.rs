use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::Serialize;
use serde_json::{json, Value};

use crate::domain::{external::Sandbox, models::ToolResult};

use super::{
    arguments::{optional_bool, optional_usize, required_non_empty_str, required_str},
    tool, BaseTool, ToolArguments, ToolDefinition,
};

const DEFAULT_MAX_LENGTH: usize = 10_000;

/// 文件工具箱：负责声明、校验并分发 Agent 可以调用的文件操作。
pub struct FileTool {
    name: String,
    sandbox: Box<dyn Sandbox>,
    definitions: Vec<ToolDefinition>,
}

impl FileTool {
    /// 创建文件工具箱，并注册允许 Agent 使用的六个文件工具。
    pub fn new(sandbox: Box<dyn Sandbox>) -> Self {
        Self {
            name: "file".to_string(),
            sandbox,
            definitions: vec![
                tool(
                    "read_file",
                    "读取文件内容。用于检查文件内容、分析日志或读取配置文件。",
                    ToolArguments::from_iter([
                        (
                            "filepath".to_string(),
                            json!({
                                "type": "string",
                                "description": "要读取文件的绝对路径"
                            }),
                        ),
                        (
                            "start_line".to_string(),
                            json!({
                                "type": "integer",
                                "description": "(可选)读取的起始行，索引从 0 开始"
                            }),
                        ),
                        (
                            "end_line".to_string(),
                            json!({
                                "type": "integer",
                                "description": "(可选)结束行号，不包含该行"
                            }),
                        ),
                        (
                            "sudo".to_string(),
                            json!({
                                "type": "boolean",
                                "description": "(可选)是否使用 sudo 权限"
                            }),
                        ),
                        (
                            "max_length".to_string(),
                            json!({
                                "type": "integer",
                                "description": "(可选)读取文件内容的最大长度，默认为 10000"
                            }),
                        ),
                    ]),
                    vec!["filepath".to_string()],
                ),
                tool(
                    "write_file",
                    "对文件进行覆盖或追加写入。用于创建新文件、追加内容或修改现有文件。",
                    ToolArguments::from_iter([
                        (
                            "filepath".to_string(),
                            json!({
                                "type": "string",
                                "description": "要写入文件的绝对路径"
                            }),
                        ),
                        (
                            "content".to_string(),
                            json!({
                                "type": "string",
                                "description": "要写入的文本内容"
                            }),
                        ),
                        (
                            "append".to_string(),
                            json!({
                                "type": "boolean",
                                "description": "(可选)是否使用追加模式"
                            }),
                        ),
                        (
                            "leading_newline".to_string(),
                            json!({
                                "type": "boolean",
                                "description": "(可选)是否在内容开头添加换行符"
                            }),
                        ),
                        (
                            "trailing_newline".to_string(),
                            json!({
                                "type": "boolean",
                                "description": "(可选)是否在内容结尾添加换行符"
                            }),
                        ),
                        (
                            "sudo".to_string(),
                            json!({
                                "type": "boolean",
                                "description": "(可选)是否使用 sudo 权限"
                            }),
                        ),
                    ]),
                    vec!["filepath".to_string(), "content".to_string()],
                ),
                tool(
                    "replace_in_file",
                    "在文件中替换指定的字符串。用于更新文件中的特定内容或修复代码中的错误。",
                    ToolArguments::from_iter([
                        (
                            "filepath".to_string(),
                            json!({
                                "type": "string",
                                "description": "要执行替换操作的文件绝对路径"
                            }),
                        ),
                        (
                            "old_str".to_string(),
                            json!({
                                "type": "string",
                                "description": "要被替换的原始字符串"
                            }),
                        ),
                        (
                            "new_str".to_string(),
                            json!({
                                "type": "string",
                                "description": "用于替换的新字符串"
                            }),
                        ),
                        (
                            "sudo".to_string(),
                            json!({
                                "type": "boolean",
                                "description": "(可选)是否使用 sudo 权限"
                            }),
                        ),
                    ]),
                    vec![
                        "filepath".to_string(),
                        "old_str".to_string(),
                        "new_str".to_string(),
                    ],
                ),
                tool(
                    "search_in_file",
                    "在文件内容中搜索匹配的文本。用于查找文件中的特定内容或模式。",
                    ToolArguments::from_iter([
                        (
                            "filepath".to_string(),
                            json!({
                                "type": "string",
                                "description": "要进行搜索的文件绝对路径"
                            }),
                        ),
                        (
                            "regex".to_string(),
                            json!({
                                "type": "string",
                                "description": "用于匹配的正则表达式模式"
                            }),
                        ),
                        (
                            "sudo".to_string(),
                            json!({
                                "type": "boolean",
                                "description": "(可选)是否使用 sudo 权限"
                            }),
                        ),
                    ]),
                    vec!["filepath".to_string(), "regex".to_string()],
                ),
                tool(
                    "find_files",
                    "在指定目录中根据名称模式查找文件。用于定位具有特定命名模式的文件。",
                    ToolArguments::from_iter([
                        (
                            "dir_path".to_string(),
                            json!({
                                "type": "string",
                                "description": "要搜索的目录绝对路径"
                            }),
                        ),
                        (
                            "glob_pattern".to_string(),
                            json!({
                                "type": "string",
                                "description": "使用 glob 语法通配符的文件名模式"
                            }),
                        ),
                    ]),
                    vec!["dir_path".to_string(), "glob_pattern".to_string()],
                ),
                tool(
                    "list_files",
                    "列出指定目录下的文件列表信息。",
                    ToolArguments::from_iter([(
                        "dir_path".to_string(),
                        json!({
                            "type": "string",
                            "description": "要列出文件列表的目录绝对路径"
                        }),
                    )]),
                    vec!["dir_path".to_string()],
                ),
            ],
        }
    }

    async fn read_file(
        &self,
        file_path: &str,
        start_line: Option<usize>,
        end_line: Option<usize>,
        sudo: Option<bool>,
        max_length: Option<usize>,
    ) -> Result<ToolResult<String>> {
        self.sandbox
            .read_file(file_path, start_line, end_line, sudo, max_length)
            .await
    }

    async fn write_file(
        &self,
        file_path: &str,
        content: &str,
        append: Option<bool>,
        leading_newline: Option<bool>,
        trailing_newline: Option<bool>,
        sudo: Option<bool>,
    ) -> Result<ToolResult<String>> {
        self.sandbox
            .write_file(
                file_path,
                content,
                append,
                leading_newline,
                trailing_newline,
                sudo,
            )
            .await
    }

    async fn replace_in_file(
        &self,
        file_path: &str,
        old_str: &str,
        new_str: &str,
        sudo: Option<bool>,
    ) -> Result<ToolResult<String>> {
        self.sandbox
            .replace_in_file(file_path, old_str, new_str, sudo)
            .await
    }

    async fn search_in_file(
        &self,
        file_path: &str,
        regex: &str,
        sudo: Option<bool>,
    ) -> Result<ToolResult<Vec<String>>> {
        self.sandbox.search_in_file(file_path, regex, sudo).await
    }

    async fn find_files(
        &self,
        dir_path: &str,
        glob_pattern: &str,
    ) -> Result<ToolResult<Vec<String>>> {
        self.sandbox.find_files(dir_path, glob_pattern).await
    }

    async fn list_files(&self, dir_path: &str) -> Result<ToolResult<Vec<String>>> {
        self.sandbox.list_files(dir_path).await
    }
}

#[async_trait]
impl BaseTool for FileTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn tool_definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    async fn call_tool(&self, tool_name: &str, kwargs: ToolArguments) -> Result<ToolResult<Value>> {
        match tool_name {
            "read_file" => {
                let file_path = required_non_empty_str(&kwargs, "filepath")?;
                let start_line = optional_usize(&kwargs, "start_line")?;
                let end_line = optional_usize(&kwargs, "end_line")?;
                let sudo = optional_bool(&kwargs, "sudo")?.or(Some(false));
                let max_length =
                    optional_usize(&kwargs, "max_length")?.or(Some(DEFAULT_MAX_LENGTH));

                into_value_result(
                    self.read_file(file_path, start_line, end_line, sudo, max_length)
                        .await?,
                )
            }
            "write_file" => {
                let file_path = required_non_empty_str(&kwargs, "filepath")?;
                let content = required_str(&kwargs, "content")?;
                let append = optional_bool(&kwargs, "append")?.or(Some(false));
                let leading_newline = optional_bool(&kwargs, "leading_newline")?.or(Some(false));
                let trailing_newline = optional_bool(&kwargs, "trailing_newline")?.or(Some(false));
                let sudo = optional_bool(&kwargs, "sudo")?.or(Some(false));

                into_value_result(
                    self.write_file(
                        file_path,
                        content,
                        append,
                        leading_newline,
                        trailing_newline,
                        sudo,
                    )
                    .await?,
                )
            }
            "replace_in_file" => {
                let file_path = required_non_empty_str(&kwargs, "filepath")?;
                let old_str = required_non_empty_str(&kwargs, "old_str")?;
                let new_str = required_str(&kwargs, "new_str")?;
                let sudo = optional_bool(&kwargs, "sudo")?.or(Some(false));

                into_value_result(
                    self.replace_in_file(file_path, old_str, new_str, sudo)
                        .await?,
                )
            }
            "search_in_file" => {
                let file_path = required_non_empty_str(&kwargs, "filepath")?;
                let regex = required_non_empty_str(&kwargs, "regex")?;
                let sudo = optional_bool(&kwargs, "sudo")?.or(Some(false));

                into_value_result(self.search_in_file(file_path, regex, sudo).await?)
            }
            "find_files" => {
                let dir_path = required_non_empty_str(&kwargs, "dir_path")?;
                let glob_pattern = required_non_empty_str(&kwargs, "glob_pattern")?;

                into_value_result(self.find_files(dir_path, glob_pattern).await?)
            }
            "list_files" => {
                let dir_path = required_non_empty_str(&kwargs, "dir_path")?;

                into_value_result(self.list_files(dir_path).await?)
            }
            _ => Err(anyhow!("工具[{tool_name}]未找到")),
        }
    }
}

fn into_value_result<T: Serialize>(result: ToolResult<T>) -> Result<ToolResult<Value>> {
    Ok(ToolResult {
        success: result.success,
        message: result.message,
        data: result.data.map(serde_json::to_value).transpose()?,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use anyhow::bail;
    use serde_json::Map;

    use super::*;
    use crate::domain::external::Browser;

    type Calls = Arc<Mutex<Vec<FileCall>>>;

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum FileCall {
        Read {
            file_path: String,
            start_line: Option<usize>,
            end_line: Option<usize>,
            sudo: Option<bool>,
            max_length: Option<usize>,
        },
        Write {
            file_path: String,
            content: String,
            append: Option<bool>,
            leading_newline: Option<bool>,
            trailing_newline: Option<bool>,
            sudo: Option<bool>,
        },
        Replace {
            file_path: String,
            old_str: String,
            new_str: String,
            sudo: Option<bool>,
        },
        Search {
            file_path: String,
            regex: String,
            sudo: Option<bool>,
        },
        Find {
            dir_path: String,
            glob_pattern: String,
        },
        List {
            dir_path: String,
        },
    }

    struct MockSandbox {
        calls: Calls,
    }

    impl MockSandbox {
        fn push(&self, call: FileCall) {
            self.calls.lock().unwrap().push(call);
        }

        fn string_result(&self, call: FileCall, data: &str) -> ToolResult<String> {
            self.push(call);
            ToolResult {
                data: Some(data.to_string()),
                ..ToolResult::default()
            }
        }

        fn list_result(&self, call: FileCall, data: &str) -> ToolResult<Vec<String>> {
            self.push(call);
            ToolResult {
                data: Some(vec![data.to_string()]),
                ..ToolResult::default()
            }
        }
    }

    #[async_trait]
    impl Sandbox for MockSandbox {
        async fn exec_command(
            &self,
            _session_id: &str,
            _exec_dir: &str,
            _command: &str,
        ) -> Result<ToolResult<String>> {
            bail!("exec_command should not be called by FileTool")
        }

        async fn read_shell_output(
            &self,
            _session_id: &str,
            _console: Option<bool>,
        ) -> Result<ToolResult<String>> {
            bail!("read_shell_output should not be called by FileTool")
        }

        async fn wait_process(
            &self,
            _session_id: &str,
            _seconds: Option<usize>,
        ) -> Result<ToolResult<String>> {
            bail!("wait_process should not be called by FileTool")
        }

        async fn write_shell_input(
            &self,
            _session_id: &str,
            _input_text: &str,
            _press_enter: Option<bool>,
        ) -> Result<ToolResult<String>> {
            bail!("write_shell_input should not be called by FileTool")
        }

        async fn kill_process(&self, _session_id: &str) -> Result<ToolResult<String>> {
            bail!("kill_process should not be called by FileTool")
        }

        async fn write_file(
            &self,
            file_path: &str,
            content: &str,
            append: Option<bool>,
            leading_newline: Option<bool>,
            trailing_newline: Option<bool>,
            sudo: Option<bool>,
        ) -> Result<ToolResult<String>> {
            Ok(self.string_result(
                FileCall::Write {
                    file_path: file_path.to_string(),
                    content: content.to_string(),
                    append,
                    leading_newline,
                    trailing_newline,
                    sudo,
                },
                "write",
            ))
        }

        async fn read_file(
            &self,
            file_path: &str,
            start_line: Option<usize>,
            end_line: Option<usize>,
            sudo: Option<bool>,
            max_length: Option<usize>,
        ) -> Result<ToolResult<String>> {
            if file_path == "/backend-error" {
                bail!("sandbox read failed");
            }

            Ok(self.string_result(
                FileCall::Read {
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    sudo,
                    max_length,
                },
                "read",
            ))
        }

        async fn check_file_exists(&self, _file_path: &str) -> Result<ToolResult<bool>> {
            bail!("check_file_exists should not be called by FileTool")
        }

        async fn delete_file(&self, _file_path: &str) -> Result<ToolResult<String>> {
            bail!("delete_file should not be called by FileTool")
        }

        async fn list_files(&self, dir_path: &str) -> Result<ToolResult<Vec<String>>> {
            let call = FileCall::List {
                dir_path: dir_path.to_string(),
            };
            if dir_path == "/failure" {
                self.push(call);
                return Ok(ToolResult {
                    success: false,
                    message: Some("list failed".to_string()),
                    data: Some(vec!["partial".to_string()]),
                });
            }

            Ok(self.list_result(call, "list"))
        }

        async fn replace_in_file(
            &self,
            file_path: &str,
            old_str: &str,
            new_str: &str,
            sudo: Option<bool>,
        ) -> Result<ToolResult<String>> {
            Ok(self.string_result(
                FileCall::Replace {
                    file_path: file_path.to_string(),
                    old_str: old_str.to_string(),
                    new_str: new_str.to_string(),
                    sudo,
                },
                "replace",
            ))
        }

        async fn search_in_file(
            &self,
            file_path: &str,
            regex: &str,
            sudo: Option<bool>,
        ) -> Result<ToolResult<Vec<String>>> {
            Ok(self.list_result(
                FileCall::Search {
                    file_path: file_path.to_string(),
                    regex: regex.to_string(),
                    sudo,
                },
                "search",
            ))
        }

        async fn find_files(
            &self,
            dir_path: &str,
            glob_pattern: &str,
        ) -> Result<ToolResult<Vec<String>>> {
            Ok(self.list_result(
                FileCall::Find {
                    dir_path: dir_path.to_string(),
                    glob_pattern: glob_pattern.to_string(),
                },
                "find",
            ))
        }

        async fn upload_file(
            &self,
            _file_data: Vec<u8>,
            _file_path: &str,
            _file_name: Option<&str>,
        ) -> Result<ToolResult<String>> {
            bail!("upload_file should not be called by FileTool")
        }

        async fn download_file(&self, _file_path: &str) -> Result<Vec<u8>> {
            bail!("download_file should not be called by FileTool")
        }

        async fn ensure_sandbox(&self) -> Result<bool> {
            bail!("ensure_sandbox should not be called by FileTool")
        }

        async fn destroy(&self) -> Result<bool> {
            bail!("destroy should not be called by FileTool")
        }

        async fn get_browser(&self) -> Result<Box<dyn Browser>> {
            bail!("get_browser should not be called by FileTool")
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

    fn file_tool() -> (FileTool, Calls) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let sandbox = MockSandbox {
            calls: Arc::clone(&calls),
        };
        (FileTool::new(Box::new(sandbox)), calls)
    }

    #[test]
    fn exposes_only_safe_file_tools_with_correct_schemas() {
        let (tool, _) = file_tool();

        assert_eq!(tool.name(), "file");
        let tools = tool.get_tools();
        let tool_names = tools
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            tool_names,
            vec![
                "read_file",
                "write_file",
                "replace_in_file",
                "search_in_file",
                "find_files",
                "list_files",
            ]
        );
        for unavailable in [
            "check_file_exists",
            "delete_file",
            "upload_file",
            "download_file",
        ] {
            assert!(!tool.has_tool(unavailable));
        }
        assert_eq!(
            tools[0]["function"]["parameters"]["required"],
            json!(["filepath"])
        );
        assert_eq!(
            tools[0]["function"]["parameters"]["properties"]["max_length"]["type"],
            "integer"
        );
        assert_eq!(
            tools[2]["function"]["parameters"]["properties"]["sudo"]["type"],
            "boolean"
        );
        assert_eq!(
            tools[4]["function"]["parameters"]["required"],
            json!(["dir_path", "glob_pattern"])
        );
    }

    #[tokio::test]
    async fn dispatches_all_file_tools_and_serializes_results() {
        let (tool, calls) = file_tool();

        let read = tool
            .invoke(
                "read_file",
                Map::from_iter([
                    ("filepath".to_string(), json!("/workspace/app.log")),
                    ("start_line".to_string(), json!(2)),
                    ("end_line".to_string(), json!(8)),
                    ("sudo".to_string(), json!(true)),
                    ("max_length".to_string(), json!(2048)),
                    ("ignored".to_string(), json!("value")),
                ]),
            )
            .await
            .unwrap();
        let write = tool
            .invoke(
                "write_file",
                Map::from_iter([
                    ("filepath".to_string(), json!("/workspace/app.txt")),
                    ("content".to_string(), json!("hello")),
                    ("append".to_string(), json!(true)),
                    ("leading_newline".to_string(), json!(false)),
                    ("trailing_newline".to_string(), json!(true)),
                    ("sudo".to_string(), json!(true)),
                ]),
            )
            .await
            .unwrap();
        let replace = tool
            .invoke(
                "replace_in_file",
                Map::from_iter([
                    ("filepath".to_string(), json!("/workspace/app.txt")),
                    ("old_str".to_string(), json!("hello")),
                    ("new_str".to_string(), json!("world")),
                    ("sudo".to_string(), json!(true)),
                ]),
            )
            .await
            .unwrap();
        let search = tool
            .invoke(
                "search_in_file",
                Map::from_iter([
                    ("filepath".to_string(), json!("/workspace/app.txt")),
                    ("regex".to_string(), json!("w.rld")),
                    ("sudo".to_string(), json!(true)),
                ]),
            )
            .await
            .unwrap();
        let find = tool
            .invoke(
                "find_files",
                Map::from_iter([
                    ("dir_path".to_string(), json!("/workspace")),
                    ("glob_pattern".to_string(), json!("**/*.rs")),
                ]),
            )
            .await
            .unwrap();
        let list = tool
            .invoke(
                "list_files",
                Map::from_iter([("dir_path".to_string(), json!("/workspace"))]),
            )
            .await
            .unwrap();

        assert_eq!(read.data, Some(json!("read")));
        assert_eq!(write.data, Some(json!("write")));
        assert_eq!(replace.data, Some(json!("replace")));
        assert_eq!(search.data, Some(json!(["search"])));
        assert_eq!(find.data, Some(json!(["find"])));
        assert_eq!(list.data, Some(json!(["list"])));
        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                FileCall::Read {
                    file_path: "/workspace/app.log".to_string(),
                    start_line: Some(2),
                    end_line: Some(8),
                    sudo: Some(true),
                    max_length: Some(2048),
                },
                FileCall::Write {
                    file_path: "/workspace/app.txt".to_string(),
                    content: "hello".to_string(),
                    append: Some(true),
                    leading_newline: Some(false),
                    trailing_newline: Some(true),
                    sudo: Some(true),
                },
                FileCall::Replace {
                    file_path: "/workspace/app.txt".to_string(),
                    old_str: "hello".to_string(),
                    new_str: "world".to_string(),
                    sudo: Some(true),
                },
                FileCall::Search {
                    file_path: "/workspace/app.txt".to_string(),
                    regex: "w.rld".to_string(),
                    sudo: Some(true),
                },
                FileCall::Find {
                    dir_path: "/workspace".to_string(),
                    glob_pattern: "**/*.rs".to_string(),
                },
                FileCall::List {
                    dir_path: "/workspace".to_string(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn applies_defaults_at_the_file_tool_boundary() {
        let (tool, calls) = file_tool();

        tool.invoke(
            "read_file",
            Map::from_iter([("filepath".to_string(), json!("/workspace/read.txt"))]),
        )
        .await
        .unwrap();
        tool.invoke(
            "write_file",
            Map::from_iter([
                ("filepath".to_string(), json!("/workspace/write.txt")),
                ("content".to_string(), json!("")),
            ]),
        )
        .await
        .unwrap();
        tool.invoke(
            "replace_in_file",
            Map::from_iter([
                ("filepath".to_string(), json!("/workspace/replace.txt")),
                ("old_str".to_string(), json!("remove me")),
                ("new_str".to_string(), json!("")),
            ]),
        )
        .await
        .unwrap();
        tool.invoke(
            "search_in_file",
            Map::from_iter([
                ("filepath".to_string(), json!("/workspace/search.txt")),
                ("regex".to_string(), json!("TODO")),
            ]),
        )
        .await
        .unwrap();

        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                FileCall::Read {
                    file_path: "/workspace/read.txt".to_string(),
                    start_line: None,
                    end_line: None,
                    sudo: Some(false),
                    max_length: Some(DEFAULT_MAX_LENGTH),
                },
                FileCall::Write {
                    file_path: "/workspace/write.txt".to_string(),
                    content: String::new(),
                    append: Some(false),
                    leading_newline: Some(false),
                    trailing_newline: Some(false),
                    sudo: Some(false),
                },
                FileCall::Replace {
                    file_path: "/workspace/replace.txt".to_string(),
                    old_str: "remove me".to_string(),
                    new_str: String::new(),
                    sudo: Some(false),
                },
                FileCall::Search {
                    file_path: "/workspace/search.txt".to_string(),
                    regex: "TODO".to_string(),
                    sudo: Some(false),
                },
            ]
        );
    }

    #[tokio::test]
    async fn rejects_missing_or_invalid_file_arguments() {
        let (tool, _) = file_tool();

        let error = tool
            .invoke(
                "find_files",
                Map::from_iter([("dir_path".to_string(), json!("/workspace"))]),
            )
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "工具参数[glob_pattern]缺失");

        let error = tool
            .invoke(
                "read_file",
                Map::from_iter([
                    ("filepath".to_string(), json!("/workspace/app.txt")),
                    ("max_length".to_string(), json!(-1)),
                ]),
            )
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "工具参数[max_length]必须是非负整数");

        let error = tool
            .invoke(
                "write_file",
                Map::from_iter([
                    ("filepath".to_string(), json!("/workspace/app.txt")),
                    ("content".to_string(), json!("hello")),
                    ("append".to_string(), json!("yes")),
                ]),
            )
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "工具参数[append]必须是布尔值");

        let error = tool
            .invoke(
                "list_files",
                Map::from_iter([("dir_path".to_string(), json!("   "))]),
            )
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "工具参数[dir_path]缺失");
    }

    #[tokio::test]
    async fn preserves_sandbox_failures_and_errors() {
        let (tool, _) = file_tool();

        let failure = tool
            .invoke(
                "list_files",
                Map::from_iter([("dir_path".to_string(), json!("/failure"))]),
            )
            .await
            .unwrap();
        assert!(!failure.success);
        assert_eq!(failure.message.as_deref(), Some("list failed"));
        assert_eq!(failure.data, Some(json!(["partial"])));

        let error = tool
            .invoke(
                "read_file",
                Map::from_iter([("filepath".to_string(), json!("/backend-error"))]),
            )
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "sandbox read failed");

        let error = tool.invoke("delete_file", Map::new()).await.unwrap_err();
        assert_eq!(error.to_string(), "工具[delete_file]未找到");
    }
}
